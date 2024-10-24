import os
import numpy as np
import nibabel as nib
import argparse
from tqdm import tqdm
import glob
from skimage.transform import resize  # Import resize function for resampling
from scipy.ndimage import zoom  # Import zoom function for resampling

# Import CRF libraries
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

def crf_slice_by_slice(image, probabilities, args):
    """
    Apply CRF post-processing to 3D medical images by processing each slice individually.
    """
    C, D, H_prob, W_prob = probabilities.shape  # Get dimensions from probabilities
    crf_predictions = []

    for i in range(D):
        prob_slice = probabilities[:, i, :, :]  # Shape: (C, H_prob, W_prob)
        img_slice = image[i, :, :]  # Shape: (H_img, W_img)

        # Resample img_slice to match prob_slice dimensions
        img_slice_resized = resize(
            img_slice,
            (H_prob, W_prob),
            order=1,  # Linear interpolation
            mode='constant',
            cval=0,
            anti_aliasing=False,
            preserve_range=True
        )
        img_slice_resized = img_slice_resized.astype(np.uint8)

        # Add a channel dimension to img_slice_resized
        img_slice_resized = img_slice_resized[:, :, np.newaxis]  # Shape: (H_prob, W_prob, 1)

        # Get H and W from img_slice_resized
        H, W = img_slice_resized.shape[:2]

        # Validate that prob_slice has the same spatial dimensions
        if prob_slice.shape[1:] != (H, W):
            print(f"Shape mismatch at slice {i}: prob_slice {prob_slice.shape}, img_slice {img_slice_resized.shape}")
            continue  # Skip this slice or handle the mismatch appropriately

        # Apply 2D CRF on this slice
        d = dcrf.DenseCRF2D(W, H, C)

        U = unary_from_softmax(prob_slice)
        d.setUnaryEnergy(U)

        # Add pairwise Gaussian
        feats = create_pairwise_gaussian(sdims=(args.sxy_gaussian, args.sxy_gaussian), shape=(H, W))
        d.addPairwiseEnergy(feats, compat=args.compat_gaussian)

        # Add pairwise Bilateral
        feats = create_pairwise_bilateral(
            sdims=(args.sxy_bilateral, args.sxy_bilateral),
            schan=(args.schan_bilateral,),
            img=img_slice_resized,
            chdim=2  # Channel dimension is now at index 2
        )
        d.addPairwiseEnergy(feats, compat=args.compat_bilateral)

        # Run inference
        Q = d.inference(args.num_iterations)
        map_slice = np.argmax(Q, axis=0).reshape((H, W))

        crf_predictions.append(map_slice)

    crf_predictions = np.stack(crf_predictions, axis=0)  # Shape: (D, H, W)
    return crf_predictions

def main():
    parser = argparse.ArgumentParser(description='Apply CRF post-processing to nnUNet predictions.')
    parser.add_argument('-i', '--images_dir', type=str, required=True,
                        help='Path to the input images directory (imagesTs).')
    parser.add_argument('-p', '--probabilities_dir', type=str, required=True,
                        help='Path to the predicted probabilities directory (from nnUNet with --save_probabilities).')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Path to the output directory where CRF-processed segmentations will be saved.')
    parser.add_argument('--gt_folder', type=str, required=True,
                        help='Path to the ground truth labels directory (labelsTs).')  # Added argument
    parser.add_argument('--num_iterations', type=int, default=5,
                        help='Number of CRF iterations.')
    parser.add_argument('--sxy_gaussian', type=float, default=3.0,
                        help='Pairwise Gaussian sxy.')
    parser.add_argument('--compat_gaussian', type=float, default=3.0,
                        help='Pairwise Gaussian compatibility.')
    parser.add_argument('--sxy_bilateral', type=float, default=80.0,
                        help='Pairwise Bilateral sxy.')
    parser.add_argument('--schan_bilateral', type=float, default=13.0,
                        help='Pairwise Bilateral schan.')
    parser.add_argument('--compat_bilateral', type=float, default=10.0,
                        help='Pairwise Bilateral compatibility.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # List of cases based on the .npz files in probabilities_dir
    cases = [os.path.splitext(f)[0] for f in os.listdir(args.probabilities_dir) if f.endswith('.npz')]

    for case in tqdm(cases):
        # Adjusted to append '_0000' to the case name
        image_file = os.path.join(args.images_dir, case + '_0000.nii.gz')

        # If the image file does not exist, try to find it using glob
        if not os.path.isfile(image_file):
            pattern = os.path.join(args.images_dir, case + '*.nii.gz')
            image_files = glob.glob(pattern)
            if len(image_files) == 0:
                print(f"Image file not found: {image_file}")
                continue
            elif len(image_files) > 1:
                print(f"Multiple image files found for case {case} in {args.images_dir}. Using the first one.")
            image_file = image_files[0]

        # Load image
        image_nifti = nib.load(image_file)
        image = image_nifti.get_fdata()

        # Load probabilities
        prob_file = os.path.join(args.probabilities_dir, case + '.npz')
        prob_data = np.load(prob_file)
        probabilities = prob_data['probabilities']

        # Ensure the image and probabilities are in the correct format
        # Get target shape from probabilities
        C, D, H_prob, W_prob = probabilities.shape

        # Resample image to match probabilities
        image_resampled = resize(
            image,
            (D, H_prob, W_prob),
            order=1,  # Linear interpolation
            mode='constant',
            cval=0,
            anti_aliasing=False,
            preserve_range=True
        )
        image_resampled = image_resampled.astype(np.float32)

        # Normalize and convert image to uint8
        image_resampled = (image_resampled - image_resampled.min()) / (image_resampled.max() - image_resampled.min()) * 255.0
        image_resampled = image_resampled.astype(np.uint8)

        # Apply CRF slice by slice
        crf_segmentation = crf_slice_by_slice(image_resampled, probabilities, args)  # Shape: (D, H_prob, W_prob)

        # Load ground truth mask to get its affine and shape
        gt_mask_file = os.path.join(args.gt_folder, case + '.nii.gz')
        if os.path.isfile(gt_mask_file):
            gt_nifti = nib.load(gt_mask_file)
            gt_data = gt_nifti.get_fdata()
            gt_shape = gt_data.shape  # Shape: (D_gt, H_gt, W_gt)
            affine = gt_nifti.affine
        else:
            # Use the image affine if ground truth is not available
            gt_shape = crf_segmentation.shape
            affine = image_nifti.affine

        # Resample crf_segmentation to match gt_shape if necessary
        if crf_segmentation.shape != gt_shape:
            # Compute scale factors
            scale_factors = [gt_dim / crf_dim for gt_dim, crf_dim in zip(gt_shape, crf_segmentation.shape)]
            crf_segmentation_resampled = zoom(crf_segmentation, zoom=scale_factors, order=0)  # Nearest neighbor interpolation
        else:
            crf_segmentation_resampled = crf_segmentation

        # Expand dimensions to include channel dimension
        crf_segmentation_resampled = np.expand_dims(crf_segmentation_resampled, axis=0)  # Shape: (1, D, H, W)

        # Save CRF-processed segmentation using the ground truth affine
        crf_nifti = nib.Nifti1Image(crf_segmentation_resampled.astype(np.uint8), affine=affine)
        nib.save(crf_nifti, os.path.join(args.output_dir, case + '.nii.gz'))

if __name__ == '__main__':
    main()