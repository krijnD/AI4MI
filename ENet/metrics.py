from collections import defaultdict

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from skimage.segmentation import find_boundaries
import math
from medpy.metric.binary import hd


def compute_hausdorff_distance(pred, gt):
    """
    Computes the adjusted Hausdorff distance between predicted segmentation and ground truth for each class.
    Handles cases where segments are missing in either prediction or ground truth.
    Penalties are scaled based on the size of the missing or spurious segments.

    Args:
        pred (torch.Tensor): Predicted segmentation tensor of shape (1, C, H, W)
        gt (torch.Tensor): Ground truth segmentation tensor of shape (1, C, H, W)

    Returns:
        list of floats: Adjusted Hausdorff distance for each class
    """
    num_classes = pred.shape[1]
    height, width = pred.shape[2], pred.shape[3]
    max_distance = math.hypot(height, width)  # Maximum possible distance (image diagonal)
    adjusted_hausdorff_distances = []

    for c in range(num_classes):
        # Get the predicted and ground truth masks for class c
        pred_mask = pred[0, c, :, :].cpu().numpy().astype(np.uint8)
        gt_mask = gt[0, c, :, :].cpu().numpy().astype(np.uint8)

        # Ensure binary masks (0 or 1)
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        gt_mask = (gt_mask > 0.5).astype(np.uint8)

        # Calculate the areas (number of pixels) of the masks
        pred_area = np.sum(pred_mask)
        gt_area = np.sum(gt_mask)
        total_area = height * width

        # Compute scaling factors
        gt_scaling_factor = gt_area / total_area
        pred_scaling_factor = pred_area / total_area

        # Extract boundary pixels
        pred_boundary = find_boundaries(pred_mask, mode='inner').astype(np.uint8)
        gt_boundary = find_boundaries(gt_mask, mode='inner').astype(np.uint8)

        # Get coordinates of boundary pixels
        pred_coords = np.column_stack(np.nonzero(pred_boundary))
        gt_coords = np.column_stack(np.nonzero(gt_boundary))

        # Determine the appropriate penalty based on the presence of segments
        if pred_coords.size == 0 and gt_coords.size == 0:
            # Both boundaries are empty; perfect agreement
            hausdorff_distance = 0.0
        elif pred_coords.size == 0 and gt_coords.size > 0:
            # GT segment is present, prediction is missing (False Negative)
            # Penalty proportional to GT segment size
            hausdorff_distance = max_distance * gt_scaling_factor
        elif pred_coords.size > 0 and gt_coords.size == 0:
            # Prediction segment is present, GT is missing (False Positive)
            # Penalty proportional to predicted segment size
            hausdorff_distance = max_distance * pred_scaling_factor
        else:
            # Both segments are present; compute the Hausdorff distance
            hd_forward = directed_hausdorff(pred_coords, gt_coords)[0]
            hd_backward = directed_hausdorff(gt_coords, pred_coords)[0]
            hausdorff_distance = max(hd_forward, hd_backward)

        adjusted_hausdorff_distances.append(hausdorff_distance)

    return adjusted_hausdorff_distances


def three_dimensional_metrics(volume_predictions, volume_ground_truths):
    metrics = defaultdict(list)
    for volume_id in volume_predictions.keys():
        # Sort slices by slice_idx
        pred_slices = sorted(volume_predictions[volume_id], key=lambda x: x[0])
        gt_slices = sorted(volume_ground_truths[volume_id], key=lambda x: x[0])

        # Stack slices to form 3D volumes
        pred_volume = torch.stack([slice_data for idx, slice_data in pred_slices], dim=1)  # Stack along depth (dim=1)
        gt_volume = torch.stack([slice_data for idx, slice_data in gt_slices], dim=1)

        # Compute 3D metrics
        dice_scores = dice_coef_per_class(pred_volume, gt_volume)
        hausdorff_distances = compute_hausdorff_distance_per_class(pred_volume, gt_volume)

        # Store or report metrics
        metrics["3D Dice"].append(dice_scores)
        metrics["3D Hausdorff distance"].append(hausdorff_distances)

    return metrics



def dice_coef_per_class(pred, target, epsilon=1e-6):
    num_classes = pred.shape[0]
    dice_scores = []
    for c in range(num_classes):
        pred_c = pred[c].flatten()
        target_c = target[c].flatten()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2. * intersection + epsilon) / (union + epsilon)
        dice_scores.append(dice.item())
    return dice_scores


def compute_hausdorff_distance_per_class(pred, target):
    num_classes = pred.shape[0]
    hausdorff_distances = []
    for c in range(num_classes):
        pred_c = pred[c].cpu().numpy()
        target_c = target[c].cpu().numpy()
        pred_mask = pred_c > 0.5
        target_mask = target_c > 0.5
        if pred_mask.sum() > 0 and target_mask.sum() > 0:
            hd_distance = hd(pred_mask, target_mask, voxelspacing=(1.0, 1.0, 1.0))
        else:
            hd_distance = np.nan  # Undefined if one of the masks is empty
        hausdorff_distances.append(hd_distance)
    return hausdorff_distances