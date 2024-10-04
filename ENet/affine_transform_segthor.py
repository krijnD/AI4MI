import argparse
import logging
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform, shift, rotate
from pathlib import Path
import os
import shutil  # Import shutil to copy files

def parse_arguments():
    parser = argparse.ArgumentParser(description='Apply affine transformation to medical images.')
    parser.add_argument('--fixed_image', type=str, required=True, help='Path to the fixed (anchor) image.')
    parser.add_argument('--moving_image_dir', type=str, required=True, help='Directory containing moving images.')
    parser.add_argument('--transform_file', type=str, required=True, help='Path to the transformation file (.tfm).')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save transformed images.')
    parser.add_argument('--replace_index', type=int, default=2, help='Index to replace in the images.')
    return parser.parse_args()

def apply_affine_transform(moving_image, transform_file):
    # Read the transformation file
    transform = sitk.ReadTransform(transform_file)
    transform_parameters = transform.GetParameters()

    # Construct the affine matrix
    affine_matrix = np.array([
        [transform_parameters[0], transform_parameters[1], transform_parameters[2], transform_parameters[9]],
        [transform_parameters[3], transform_parameters[4], transform_parameters[5], transform_parameters[10]],
        [transform_parameters[6], transform_parameters[7], transform_parameters[8], transform_parameters[11]],
        [0, 0, 0, 1]
    ])

    rotation_scale = affine_matrix[:3, :3]
    translation = affine_matrix[:3, 3]
    moving_image_data = np.array(moving_image)

    # Apply affine transform
    transformed_image_data = affine_transform(moving_image_data, rotation_scale, offset=translation, order=0)

    # Get and apply shift
    shift_params = np.array(transform.GetFixedParameters())
    shifted_volume = shift(transformed_image_data, shift_params, order=0)

    # Apply rotation (if needed)
    # rotated_volume = rotate(shifted_volume, -21, axes=(0,1), reshape=False, order=0)

    # Shift back to original position
    # transformed_image_data = shift(rotated_volume, -shift_params, order=0)
    transformed_image_data = shifted_volume  # Use this if rotation is not applied

    return transformed_image_data

def save_image_with_nibabel(transformed_image_data, reference_image_path, output_image_path, replace_index=None):
    reference_image = nib.load(reference_image_path)
    reference_affine = reference_image.affine
    reference_header = reference_image.header

    if replace_index is not None:
        reference_image_data = reference_image.get_fdata()
        reference_image_data[reference_image_data == replace_index] = 0
        reference_image_data[transformed_image_data == 1] = replace_index
        transformed_image_data = reference_image_data

    transformed_image_nifti = nib.Nifti1Image(transformed_image_data.astype(np.uint8), reference_affine, reference_header)
    nib.save(transformed_image_nifti, output_image_path)

def process_images(args):
    fixed_image = (nib.load(args.fixed_image).get_fdata() == args.replace_index).astype(np.float32)
    moving_images = list(Path(args.moving_image_dir).rglob('**/GT.nii.gz'))

    if not moving_images:
        logging.error(f"No moving images found in directory: {args.moving_image_dir}")
        return

    # Process GT.nii.gz files
    for img_path in moving_images:
        try:
            moving_image = nib.load(img_path).get_fdata()

            if args.replace_index:
                moving_image = (moving_image == args.replace_index).astype(np.float32)

            transformed_image_data = apply_affine_transform(moving_image, args.transform_file)
            output_image_path = Path(args.output_dir) / img_path.relative_to(args.moving_image_dir)
            output_image_path.parent.mkdir(parents=True, exist_ok=True)

            # Use the original GT.nii.gz as reference
            save_image_with_nibabel(transformed_image_data.round(), str(img_path), str(output_image_path), replace_index=args.replace_index)

            logging.info(f"Transformed GT saved: {output_image_path}")
        except Exception as e:
            logging.error(f"Failed to process GT image {img_path}: {e}")

    # Copy Patient_<number>.nii.gz files to output directory
    patient_images = list(Path(args.moving_image_dir).rglob('**/Patient_*.nii.gz'))

    if not patient_images:
        logging.warning(f"No patient images found in directory: {args.moving_image_dir}")
    else:
        for img_path in patient_images:
            try:
                output_image_path = Path(args.output_dir) / img_path.relative_to(args.moving_image_dir)
                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(img_path), str(output_image_path))
                logging.info(f"Copied patient image: {output_image_path}")
            except Exception as e:
                logging.error(f"Failed to copy patient image {img_path}: {e}")

def main():
    args = parse_arguments()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Process images
    process_images(args)

if __name__ == '__main__':
    main()