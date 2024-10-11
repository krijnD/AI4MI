import os
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import cv2

def dense_crf(image, output_probs):
    """
    Apply DenseCRF to refine the ENet segmentation output.
    
    :param image: The original image (H, W, 3)
    :param output_probs: The ENet output probabilities (C, H, W) where C is the number of classes
    :return: Refined segmentation map (H, W)
    """
    # Prepare the DenseCRF model
    H, W = image.shape[:2]
    num_classes = output_probs.shape[0]
    d = dcrf.DenseCRF2D(W, H, num_classes)

    # Get unary potentials (negative log probabilities)
    unary = -np.log(output_probs)
    unary = unary.reshape((num_classes, -1))
    d.setUnaryEnergy(unary)

    # Add pairwise Gaussian potentials for spatial coherence
    d.addPairwiseGaussian(sxy=3, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Add pairwise bilateral potentials for appearance-based consistency
    d.addPairwiseBilateral(sxy=10, srgb=13, rgbim=image, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Perform inference
    Q = d.inference(5)

    # Get the most probable class for each pixel
    refined_segmentation = np.argmax(Q, axis=0).reshape((H, W))
    return refined_segmentation

# Define paths
base_path = "results/toy2/epoch5/ce/iter004/val"
probs_path = "results/toy2/epoch5/ce/iter004/probs"  # Adjust if needed
output_path = "results/toy2/epoch5/ce/iter004/refined_val"
os.makedirs(output_path, exist_ok=True)

# Loop through all PNG files in the folder
for file_name in sorted(os.listdir(base_path)):
    if file_name.endswith('.png'):
        image_path = os.path.join(base_path, file_name)
        image = cv2.imread(image_path)

        # Load the corresponding probability map
        prob_file_name = file_name.replace('.png', '.npy')
        output_probs_path = os.path.join(probs_path, prob_file_name)
        output_probs = np.load(output_probs_path)

        # Apply DenseCRF post-processing
        refined_segmentation = dense_crf(image, output_probs)

        # Save the refined output
        refined_output_path = os.path.join(output_path, file_name)
        cv2.imwrite(refined_output_path, refined_segmentation * 255)
        print(f"Refined image saved to {refined_output_path}")
