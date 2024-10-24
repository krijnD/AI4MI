import argparse
import pickle
import uuid
from collections import defaultdict
from datetime import datetime
from pprint import pprint
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from operator import itemgetter
from DenseCRF import dense_crf_from_probabilities
import cv2

# Import your custom modules (make sure these are accessible in your notebook)
from ENet import ENet
from evaluate_plots import class_names, plot_results, plot_metrics, animate_3d_volume
from metrics import compute_hausdorff_distance, dice_coef_per_class, compute_hausdorff_distance_per_class, \
    three_dimensional_metrics
from dataset import SliceDataset
from utils import dice_coef, probs2one_hot, probs2class, class2one_hot

from pathlib import Path


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--crf', action='store_true')

    # Add arguments for DenseCRF parameters
    parser.add_argument('--sxy_gaussian', type=int, default=1)
    parser.add_argument('--compat_gaussian', type=int, default=1)
    parser.add_argument('--sxy_bilateral', type=int, default=5)
    parser.add_argument('--srgb_bilateral', type=int, default=2)
    parser.add_argument('--compat_bilateral', type=int, default=1)
    parser.add_argument('--num_iterations', type=int, default=5)

    args = parser.parse_args()

    pprint(args)
    return args


def initialize_model(model_path, device):
    model = ENet(1, 5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def initialize_data_test():
    root_dir = Path("data") / "SEGTHOR_train"

    K = 5

    img_transform = transforms.Compose([
        lambda img: img.convert('L'),
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: np.clip((255 / nd.max()) * nd * 1.2, 0, 255),  # Adjust brightness
        lambda nd: nd / 255.0,
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

    # Ground truth transformations
    gt_transform = transforms.Compose([
        lambda img: np.array(img),
        lambda nd: nd / 63,  # Specific to SEGTHOR dataset; adjust if necessary
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],
        lambda t: class2one_hot(t, K=K),
        itemgetter(0)
    ])

    # Create the dataset
    test_set = SliceDataset('test', root_dir, img_transform=img_transform, gt_transform=gt_transform)

    # Create the data loader
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return test_loader


def crf_post_processing(image, probs, args):
    image_crf = image.squeeze().cpu().numpy()  # Shape: (H, W)
    image_crf = (image_crf * 255).astype(np.uint8)  # Convert to uint8
    image_crf = cv2.cvtColor(image_crf, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel RGB

    # Get probabilities for CRF
    pred_probs_crf = probs[0].cpu().numpy()  # Shape: (C, H, W)

    # Apply DenseCRF
    crf_probs = dense_crf_from_probabilities(
        image_crf, pred_probs_crf,
        sxy_gaussian=args.sxy_gaussian,
        compat_gaussian=args.compat_gaussian,
        sxy_bilateral=args.sxy_bilateral,
        srgb_bilateral=args.srgb_bilateral,
        compat_bilateral=args.compat_bilateral,
        num_iterations=args.num_iterations
    )

    crf_probs = torch.tensor(crf_probs, dtype=torch.float32).unsqueeze(0)
    crf_onehot = probs2one_hot(crf_probs)  # Convert to one-hot (for consistency with the rest of the code

    # return crf_onehot as tensor
    return torch.Tensor(crf_onehot)


def make_eval_dir(args):
    parent_dir = os.path.join("ENet", "evaluation")
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    current_time = datetime.now().strftime("%m-%d(%H.%M)")
    directory_name = f"{args.model_name}_{current_time}"
    directory_path = os.path.join(parent_dir, directory_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    print(f"Saving evaluation results to: {directory_path}")
    return directory_path


def print_metrics(metrics, evaluate_dir, name="metrics"):
    with open(os.path.join(evaluate_dir, name + ".txt"), 'w') as f:
        for name_score, values in metrics.items():
            scores = np.array(values)  # Shape: (num_samples, num_classes)
            score_per_class = scores.mean(axis=0)
            print(f"Mean {name_score} per class:")
            f.write(f"Mean {name_score} per class:\n")
            for i, score in enumerate(score_per_class):
                f.write(f"Class {class_names[i]}: {score:.4f}\n")
                print(f"Class {class_names[i]}: {score:.4f}")
            f.write(f"Mean (excluding background): {score_per_class[1:].mean()}")
            print(f"Mean (excluding background) : {score_per_class[1:].mean()}")
            f.write("\n")
            print("\n")


def main():
    args = args_parser()

    evaluate_dir = make_eval_dir(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = initialize_model(args.model_path, device).to(device)

    test_loader = initialize_data_test()

    metrics = defaultdict(list)
    volume_predictions = defaultdict(list)
    volume_ground_truths = defaultdict(list)

    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader)):
            img = data['images'].to(device)  # (B, C, H, W) -> (1, 5, 256, 256)
            gt = data['gts'].to(device)  # (B, K, H, W) ->

            volume_id = data['volume_id'].item()  # Batch size must be 1!
            slice_idx = data['slice_idx'].item()

            plottables = defaultdict(torch.Tensor)
            plottables["Ground truth"] = gt

            # Forward pass
            pred_logits = model(img)  # Shape: (B, K, H, W)
            pred_probs = F.softmax(pred_logits, dim=1)  # Shape: (B, K, H, W)

            # Convert probabilities to one-hot
            pred_one_hot = probs2one_hot(pred_probs)  # Shape: (B, K, H, W)
            plottables["Normal prediction"] = pred_one_hot

            # Save to 3D slices
            pred_slice = pred_one_hot.squeeze(0).cpu()
            gt_slice = gt.squeeze(0).cpu()
            volume_predictions[volume_id].append((slice_idx, pred_slice))
            volume_ground_truths[volume_id].append((slice_idx, gt_slice))

            if args.crf:
                crf_pred_one_hot = crf_post_processing(img, pred_probs, args).to(device)
                plottables["Dense CRF prediction"] = crf_pred_one_hot

            if batch_idx % 50 == 0:
                plot_results(img, plottables, batch_idx, evaluate_dir)

            # Add metrics

            # Compute Dice coefficients
            dice_normal = dice_coef(gt, pred_one_hot)
            metrics["Dice"].append(dice_normal[0].cpu().numpy())

            if args.crf:
                dice_crf = dice_coef(gt, crf_pred_one_hot)
                metrics["Dice CRF"].append(dice_crf[0].cpu().numpy())

            # TODO: add more metrics here in the same manner!! Then plot will take care of rest
            # metrics["name of metric"].append(value)
            # value should be a list in shape [score1, score2, ..., score5] where score1 is the score for class 1, etc.
            hausdorff = compute_hausdorff_distance(pred_one_hot, gt)
            # print(hausdorff)
            metrics["Hausdorff distance"].append(hausdorff)

            # # This is used for testing so that it doesn't take too long
            # if batch_idx > 10:
            #     break

    metrics_3d = three_dimensional_metrics(volume_predictions, volume_ground_truths)
    with open(evaluate_dir + "/metrics3D.pkl", "wb") as f:
        pickle.dump(metrics_3d, f)
    print_metrics(metrics_3d, evaluate_dir, name="metrics3D")
    plot_metrics(metrics_3d, evaluate_dir)
    animate_3d_volume(volume_predictions, volume_ground_truths, evaluate_dir)

    # make pickle object of metrics dict
    with open(evaluate_dir + "/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    print_metrics(metrics, evaluate_dir)
    plot_metrics(metrics, evaluate_dir)


if __name__ == "__main__":
    main()
