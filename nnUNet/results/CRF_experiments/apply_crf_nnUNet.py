import os
import numpy as np
import nibabel as nib
import argparse
from tqdm import tqdm

# Import CRF libraries (you might need to install pydensecrf)
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

def crf_3d(image, probabilities, args):
    """
    Apply CRF post-processing to 3D medical images.
    """
    C, *dims = probabilities.shape  # C, D, H, W or C, H, W
    use_2d = (len(dims) == 2)  # Check if the data is 2D or 3D

    if use_2d:
        H, W = dims
        d = dcrf.DenseCRF2D(W, H, C)
    else:
        D, H, W = dims
        d = dcrf.DenseCRF3D(W, H, D, C)

    # Prepare unary potentials
    U = unary_from_softmax(probabilities)
    d.setUnaryEnergy(U)

    # Add pairwise Gaussian
    if use_2d:
        feats = create_pairwise_gaussian(sdims=(args.sxy_gaussian, args.sxy_gaussian), shape=(H, W))
    else:
        feats = create_pairwise_gaussian(sdims=(args.sxy_gaussian, args.sxy_gaussian, args.sxy_gaussian), shape=(D, H, W))
    d.addPairwiseEnergy(feats, compat=args.compat_gaussian)

    # Add pairwise Bilateral
    if use_2d:
        feats = create_pairwise_bilateral(sdims=(args.sxy_bilateral, args.sxy_bilateral),
                                          schan=(args.schan_bilateral,),
                                          img=image.astype(np.uint8), chdim=2)
    else:
        feats = create_pairwise_bilateral(sdims=(args.sxy_bilateral, args.sxy_bilateral, args.sxy_bilateral),
                                          schan=(args.schan_bilateral,),
                                          img=image.astype(np.uint8), chdim=3)
    d.addPairwiseEnergy(feats, compat=args.compat_bilateral)

    # Run inference
    Q = d.inference(args.num_iterations)

    # Get the MAP prediction
    map_prediction = np.argmax(Q, axis=0)

    # Reshape to original image shape
    if use_2d:
        map_prediction = map_prediction.reshape((H, W))
    else:
        map_prediction = map_prediction.reshape((D, H, W))

    return map_prediction

def main():
    parser = argparse.ArgumentParser(description='Apply CRF post-processing to nnUNet predictions.')
    parser.add_argument('-i', '--images_dir', type=str, required=True,
                        help='Path to the input images directory (imagesTs).')
    parser.add_argument('-p', '--probabilities_dir', type=str, required=True,
                        help='Path to the predicted probabilities directory (from nnUNet with --save_probabilities).')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Path to the output directory where CRF-processed segmentations will be saved.')
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

    # List of cases
    cases = [f.replace('.npz', '') for f in os.listdir(args.probabilities_dir) if f.endswith('.npz')]

    for case in tqdm(cases):
        # Load image
        image_nifti = nib.load(os.path.join(args.images_dir, case + '.nii.gz'))
        image = image_nifti.get_fdata()

        # Load probabilities
        prob_data = np.load(os.path.join(args.probabilities_dir, case + '.npz'))
        probabilities = prob_data['probabilities']

        # Ensure the image and probabilities are in the correct format
        # Convert image to uint8 and scale if necessary
        image = (image - image.min()) / (image.max() - image.min()) * 255.0
        image = image.astype(np.uint8)

        # Apply CRF
        crf_segmentation = crf_3d(image, probabilities, args)

        # Save CRF-processed segmentation
        crf_nifti = nib.Nifti1Image(crf_segmentation.astype(np.uint8), affine=image_nifti.affine)
        nib.save(crf_nifti, os.path.join(args.output_dir, case + '.nii.gz'))

if __name__ == '__main__':
    main()