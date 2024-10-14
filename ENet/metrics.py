import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from skimage.segmentation import find_boundaries
import math


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
