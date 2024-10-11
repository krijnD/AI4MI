import os
import numpy as np
import cv2
from pathlib import Path
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


def dense_crf_from_probabilities(image, probabilities):
    """
    Apply DenseCRF to refine the segmentation probabilities.

    Parameters:
    - image: NumPy array of shape (H, W) or (H, W, 3), the original image.
    - probabilities: NumPy array of shape (C, H, W), the predicted probabilities for each class.

    Returns:
    - refined_probabilities: NumPy array of shape (C, H, W), the refined probabilities after DenseCRF.
    """
    num_classes, H, W = probabilities.shape
    d = dcrf.DenseCRF2D(W, H, num_classes)

    # Prepare unary potentials
    unary = -np.log(probabilities + 1e-8)  # Add epsilon to avoid log(0)
    unary = unary.reshape((num_classes, -1))
    d.setUnaryEnergy(unary)


    # Add pairwise potentials
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=10, srgb=13, rgbim=image, compat=10)

    # Perform inference
    Q = d.inference(5)
    refined_probabilities = np.array(Q).reshape((num_classes, H, W))

    return refined_probabilities


def main():
    print("Starting DenseCRF post-processing...")
    # Define paths
    images_dir = Path('results/toy2/epoch5/ce/iter004/val')  # Replace with your images directory
    probs_dir = Path('results/toy2/epoch5/ce/iter004/probs')  # Replace with your probabilities directory
    output_dir = Path('results/toy2/epoch5/ce/refined_probs')  # Replace with your desired output directory
    print(f"Images directory: {images_dir.resolve()}")
    print(f"Probabilities directory: {probs_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    if not images_dir.exists():
        print(f"Images directory does not exist: {images_dir}")
    if not probs_dir.exists():
        print(f"Probabilities directory does not exist: {probs_dir}")


    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_classes = 2  # Adjust based on your dataset (e.g., 2 for TOY2, 5 for SEGTHOR)
    
    # List all probability files
    prob_files = sorted(probs_dir.glob('*.npy'))
    
    for prob_file in prob_files:
        # Load the predicted probabilities
        probabilities = np.load(prob_file)  # Shape: (C, H, W)
        
        # Load the corresponding original image
        stem = prob_file.stem
        image_file = images_dir / f"{stem}.png"  # Adjust the extension if necessary
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Could not read original image: {image_file}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply DenseCRF refinement
        refined_probabilities = dense_crf_from_probabilities(image, probabilities)
        
        # Save the refined probabilities
        output_file = output_dir / f"{stem}.npy"
        np.save(output_file, refined_probabilities)
        print(f"Refined probabilities saved: {output_file}")

if __name__ == '__main__':
    main()