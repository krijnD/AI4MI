#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import CrossEntropy, DiceLoss, CombinedLoss


datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the clases with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8}


def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    net = datasets_params[args.dataset]['net'](1, K)
    net.init_weights()
    net.to(device)

    lr = 0.0005
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    root_dir = Path(__file__).resolve().parent.parent / "data" / args.dataset

    img_transform = transforms.Compose([
        lambda img: img.convert('L'),
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: np.clip((255 / nd.max()) * nd * 1.2, 0, 255), # TODO: I adjusted the brightness here by 1.2
        lambda nd: nd / 255, 
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

    gt_transform = transforms.Compose([
        lambda img: np.array(img)[...],
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # max <= 1
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
        lambda t: class2one_hot(t, K=K), # Go from (1, H, W) to (1, K, H, W) in one hot encoding for every class
        itemgetter(0)
    ])


    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform=gt_transform,
                             debug=args.debug)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=args.num_workers,
                              shuffle=True)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=gt_transform,
                           debug=args.debug)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=args.num_workers,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    model, optimizer, device, train_loader, val_loader, num_classes = setup(args)

    # Choose the loss function based on args.loss
    if args.loss == 'CombinedLoss':
        if args.mode == "full":
            loss_fn = CombinedLoss(idk=list(range(num_classes)), weight_ce=args.weight_ce, weight_dice=args.weight_dice)
        elif args.mode in ["partial"] and args.dataset in ['SEGTHOR', 'SEGTHOR_STUDENTS']:
            loss_fn = CombinedLoss(idk=[0, 1, 3, 4], weight_ce=args.weight_ce, weight_dice=args.weight_dice)
        else:
            raise ValueError(args.mode, args.dataset)
    elif args.loss == 'DiceLoss':  # Changed from 'if' to 'elif'
        if args.mode == "full":
            loss_fn = DiceLoss(idk=list(range(num_classes)))  # Supervise both background and foreground
        elif args.mode in ["partial"] and args.dataset in ['SEGTHOR', 'SEGTHOR_STUDENTS']:
            loss_fn = DiceLoss(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
        else:
            raise ValueError(args.mode, args.dataset)
    elif args.loss == 'CrossEntropy':
        if args.mode == "full":
            loss_fn = CrossEntropy(idk=list(range(num_classes)))  # Supervise both background and foreground
        elif args.mode in ["partial"] and args.dataset in ['SEGTHOR', 'SEGTHOR_STUDENTS']:
            loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
        else:
            raise ValueError(args.mode, args.dataset)
    else:
        raise ValueError(f"Unknown loss function {args.loss}")

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), num_classes))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), num_classes))

    best_dice: float = 0

    for epoch in range(args.epochs):
        for method in ['train', 'val']:
            match method:
                case 'train':
                    model.train()
                    optimizer_instance = optimizer
                    context_manager = Dcm
                    desc = f">> Training   ({epoch: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    model.eval()
                    optimizer_instance = None
                    context_manager = torch.no_grad
                    desc = f">> Validation ({epoch: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with context_manager():  # Either dummy context manager, or the torch.no_grad for validation
                global_sample_idx = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for batch_idx, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if optimizer_instance:  # So only for training
                        optimizer_instance.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    batch_size, _, W, H = img.shape

                    pred_logits = model(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[epoch, global_sample_idx:global_sample_idx + batch_size, :] = dice_coef(gt, pred_seg)  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    log_loss[epoch, batch_idx] = loss.item()  # One loss value per batch (averaged in the loss)

                    if optimizer_instance:  # Only for training
                        loss.backward()
                        optimizer_instance.step()

                    if method == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if num_classes == 5 else (255 / (num_classes - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{epoch:03d}" / method)

                    global_sample_idx += batch_size  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[epoch, :global_sample_idx, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[epoch, :batch_idx + 1].mean():5.2e}"}
                    if num_classes > 2:
                        postfix_dict |= {f"Dice-{class_idx}": f"{log_dice[epoch, :global_sample_idx, class_idx].mean():05.3f}"
                                         for class_idx in range(1, num_classes)}
                    tq_iter.set_postfix(postfix_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = log_dice_val[epoch, :, 1:].mean().item()
        if current_dice > best_dice:
            print(f">>> Improved dice at epoch {epoch}: {best_dice:05.3f}->{current_dice:05.3f} DSC")
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                    f.write(str(epoch))

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                    rmtree(best_folder)
            copytree(args.dest / f"iter{epoch:03d}", Path(best_folder))

            torch.save(model, args.dest / "bestmodel.pkl")
            torch.save(model.state_dict(), args.dest / "bestweights.pt")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logic around epochs and logging easily.")
    # Add the new --loss argument
    parser.add_argument('--loss', default='CrossEntropy', choices=['CrossEntropy', 'DiceLoss', 'CombinedLoss'],
                        help="Loss function to use during training.")
    # Add optional arguments for loss weights
    parser.add_argument('--weight_ce', type=float, default=1.0,
                        help="Weight for Cross Entropy loss in CombinedLoss.")
    parser.add_argument('--weight_dice', type=float, default=1.0,
                        help="Weight for Dice loss in CombinedLoss.")

    args = parser.parse_args()

    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()
