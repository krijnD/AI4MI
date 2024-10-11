import argparse
import uuid
from collections import defaultdict
from datetime import datetime
from pprint import pprint
import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from operator import itemgetter
from DenseCRF import dense_crf_from_probabilities
import cv2

# Import your custom modules (make sure these are accessible in your notebook)
from ENet import ENet
from dataset import SliceDataset
from utils import dice_coef, probs2one_hot, probs2class, class2one_hot

from pathlib import Path


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--crf', action='store_true')

    args = parser.parse_args()

    pprint(args)
    return args


def initialize_model(model_path, device):
    model = ENet(1, 5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def initialize_data_test():
    root_dir = Path("data") / "SEGTHOR"

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


def plot_results(image, plottables, idx, evaluate_dir):
    img_np = image.cpu().numpy()[0, 0, :, :]  # Shape: (H, W)
    fig, axs = plt.subplots(1, len(plottables) + 1, figsize=(20, 5))

    axs_i = 0
    axs[axs_i].imshow(img_np, cmap='gray')
    axs[axs_i].set_title('Input Image')
    axs[axs_i].axis('off')
    axs_i += 1

    for name, prediction in plottables.items():
        pred_class = torch.argmax(prediction, dim=1)  # Shape: (B, H, W)
        prediction_np = pred_class.cpu().numpy()[0]  # Shape: (H, W)

        axs[axs_i].imshow(prediction_np, cmap='jet', vmin=0, vmax=5 - 1)
        axs[axs_i].set_title(name)
        axs[axs_i].axis('off')

        axs_i += 1

    plt.savefig(evaluate_dir + "/predict_" + str(idx) + ".png")




def crf_post_processing(image, probs):
    image_crf = image.squeeze().cpu().numpy()  # Shape: (H, W)
    image_crf = (image_crf * 255).astype(np.uint8)  # Convert to uint8
    image_crf = cv2.cvtColor(image_crf, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel RGB

    # Get probabilities for CRF
    pred_probs_crf = probs[0].cpu().numpy()  # Shape: (C, H, W)

    # Apply DenseCRF
    crf_probs = dense_crf_from_probabilities(image_crf, pred_probs_crf)
    crf_probs = torch.tensor(crf_probs, dtype=torch.float32).unsqueeze(0)
    crf_onehot = probs2one_hot(crf_probs)  # Convert to one-hot (for consistency with the rest of the code

    return crf_onehot


def make_eval_dir(args):
    parent_dir = os.path.join("ENet", "evaluation")
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    current_time = datetime.now().strftime("%m-%d(%H.%M)")
    directory_name = f"evaluation_{args.model_name}_{current_time}"
    directory_path = os.path.join(parent_dir, directory_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    print(f"Saving evaluation results to: {directory_path}")
    return directory_path


def main():
    args = args_parser()

    evaluate_dir = make_eval_dir(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = initialize_model(args.model_path, device)

    test_loader = initialize_data_test()

    metrics = defaultdict(list)

    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader)):
            img = data['images'].to(device) #(B, C, H, W) -> (1, 5, 256, 256)
            gt = data['gts'].to(device) #(B, K, H, W) ->
            plottables = defaultdict(torch.Tensor)
            plottables["Ground truth"] = gt

            # Forward pass
            pred_logits = model(img) # Shape: (B, K, H, W)
            pred_probs = F.softmax(pred_logits, dim=1) # Shape: (B, K, H, W)

            # Convert probabilities to one-hot
            pred_one_hot = probs2one_hot(pred_probs) # Shape: (B, K, H, W)
            plottables["Normal prediction"] = pred_one_hot

            if args.crf:
                crf_pred_one_hot = crf_post_processing(img, pred_probs)
                plottables["Dense CRF prediction"] = crf_pred_one_hot


            if batch_idx % 50 == 0:
                plot_results(img, plottables, batch_idx, evaluate_dir)


            # Add metrics
            # Compute Dice coefficient
            dice_normal = dice_coef(gt, pred_one_hot)
            metrics["dice_normal"].append(dice_normal[0].cpu().numpy())

            if args.crf:
                dice_crf = dice_coef(gt, crf_pred_one_hot)
                metrics["dice_crf"].append(dice_crf[0].cpu().numpy())

            # TODO: add more metrics here in the same manner!! Then plot will take care of rest

            # if batch_idx > 10:
            #     break


    # Save metrics
    class_names = ['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta']
    with open(os.path.join(evaluate_dir, "metrics.txt"), 'w') as f:
        for name_score, values in metrics.items():

            # Print and write scores
            scores = np.array(values)
            score_per_class = scores.mean(axis=0)
            print(f"Mean {name_score} per class:")
            f.write(f"Mean {name_score} per class:\n")
            for i, score in enumerate(score_per_class):
                f.write(f"Class {class_names[i]}: {score}\n")
                print(f"Class {class_names[i]}: {score}")
            f.write("\n")
            print("\n")

            # Create boxplot
            data_to_plot = [scores[:, i] for i in range(len(class_names))]

            plt.figure()
            plt.boxplot(data_to_plot, patch_artist=True)
            plt.xticks(range(1, len(class_names) + 1), class_names, rotation=45)
            plt.title(f"{name_score} per class")
            plt.ylabel(name_score)
            plt.tight_layout()
            plt.savefig(os.path.join(evaluate_dir, f"{name_score}_per_class.png"))
            plt.close()

if __name__ == "__main__":
    main()
