import SimpleITK as sitk
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform, shift, rotate
from pathlib import Path

# Paths for fixed (anchor) and moving (distorted) images
fixed_image_path = 'data/segthor_train/train/Patient_27/GT2.nii.gz'
moving_image_path = 'data/segthor_train/train/Patient_27/GT.nii.gz'
replace_index = 2


def apply_affine_transform(moving_image, transform_file):
    transform = sitk.ReadTransform(transform_file)
    transform_parameters = transform.GetParameters()

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

    # Apply rotation
    rotated_volume = rotate(shifted_volume, -21, axes=(0,1), reshape=False, order=0)

    # Shift back to original position
    transformed_image_data = shift(rotated_volume, -shift_params, order=0)

    return transformed_image_data

def save_image_with_nibabel(transformed_image_data, reference_image_path, output_image_path, replace_index=None):
    reference_image = nib.load(reference_image_path)
    reference_affine = reference_image.affine
    reference_header = reference_image.header

    if replace_index:
        reference_image_data = reference_image.get_fdata()
        reference_image_data[reference_image_data == replace_index] = 0
        reference_image_data[transformed_image_data == 1] = replace_index
        transformed_image_data = reference_image_data

    transformed_image_nifti = nib.Nifti1Image(transformed_image_data.astype(np.uint8), reference_affine, reference_header)
    nib.save(transformed_image_nifti, output_image_path)

# Main script execution
fixed_image = (nib.load(fixed_image_path).get_fdata() == replace_index).astype(np.float32)
moving_image = (nib.load(moving_image_path).get_fdata() == replace_index).astype(np.float32)
output_image_path = 'transformed_image.nii.gz'

# Apply the saved affine transformation
transformed_image_data = apply_affine_transform(moving_image, 'transform.tfm')

# Save the transformed image
save_image_with_nibabel(transformed_image_data.round(), moving_image_path, output_image_path, replace_index)
print(f"Final overlap ratio: {(transformed_image_data == fixed_image).sum() / fixed_image.size:.4f}")

# Process all the images in the dataset
for img in Path.cwd().rglob('**/GT.nii.gz'):
    moving_image = nib.load(img).get_fdata()

    # Rename original GT to GT_old
    old_img_path = img.with_name("GT_old.nii.gz")
    img.rename(old_img_path)
    
    if replace_index:
        moving_image = (moving_image == replace_index).astype(np.float32)

    # Apply affine transformation and save the transformed image
    transformed_image_data = apply_affine_transform(moving_image, 'transform.tfm')
    output_image_path = img.with_name("GT.nii.gz")
    save_image_with_nibabel(transformed_image_data.round(), old_img_path, output_image_path, replace_index=replace_index)
    # Print confirmation that the new GT has been made
    print(f"New GT created: {output_image_path}")
