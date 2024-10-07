import SimpleITK as sitk
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform, shift, rotate
from pathlib import Path
import shutil
import os

# Define root directories for ENet and nnU-Net data
root_dir = Path.cwd().parent
print(f"Root directory: {root_dir}")

enet_dir = root_dir / 'ENet_data' / 'original_data'
nnunet_dir = root_dir / 'nnU-Net_data' / 'original_data'

# Set paths for fixed (anchor) and moving (distorted) images
replace_index = 2

def apply_affine_transform(moving_image, transform_file):
    print("Reading transformation file...")
    try:
        transform = sitk.ReadTransform(transform_file)
    except Exception as e:
        print(f"Error reading transformation file: {transform_file}. Error: {e}")
        return None

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

    try:
        # Apply affine transform
        print("Applying affine transformation...")
        transformed_image_data = affine_transform(moving_image_data, rotation_scale, offset=translation, order=0)

        # Get and apply shift
        print("Applying shift transformation...")
        shift_params = np.array(transform.GetFixedParameters())
        shifted_volume = shift(transformed_image_data, shift_params, order=0)

        # Apply rotation
        print("Applying rotation...")
        rotated_volume = rotate(shifted_volume, -21, axes=(0, 1), reshape=False, order=0)

        # Shift back to original position
        print("Shifting back to original position...")
        transformed_image_data = shift(rotated_volume, -shift_params, order=0)
    except Exception as e:
        print(f"Error applying affine transformation. Error: {e}")
        return None

    return transformed_image_data

def save_image_with_nibabel(transformed_image_data, reference_image_path, output_image_path, replace_index=None):
    print(f"Saving transformed image to {output_image_path}...")
    try:
        reference_image = nib.load(reference_image_path)
    except Exception as e:
        print(f"Error loading reference image: {reference_image_path}. Error: {e}")
        return

    reference_affine = reference_image.affine
    reference_header = reference_image.header

    if replace_index:
        reference_image_data = reference_image.get_fdata()
        reference_image_data[reference_image_data == replace_index] = 0
        reference_image_data[transformed_image_data == 1] = replace_index
        transformed_image_data = reference_image_data

    try:
        transformed_image_nifti = nib.Nifti1Image(transformed_image_data.astype(np.uint8), reference_affine, reference_header)
        nib.save(transformed_image_nifti, output_image_path)
        print(f"Saved {output_image_path} successfully.")
    except Exception as e:
        print(f"Error saving transformed image: {output_image_path}. Error: {e}")

# Create patient directories if they don't exist
def create_patient_directories(patient_number):
    print(f"Creating directories for Patient {patient_number}...")
    enet_patient_dir = enet_dir / f'Patient_{patient_number}'
    nnunet_patient_dir = nnunet_dir / f'Patient_{patient_number}'

    enet_patient_dir.mkdir(parents=True, exist_ok=True)
    nnunet_patient_dir.mkdir(parents=True, exist_ok=True)

    return enet_patient_dir, nnunet_patient_dir

# Copy GT.nii.gz to ENet and nnU-Net folders
def copy_gt_to_enet_nnunet(new_gt_path, enet_patient_dir, nnunet_patient_dir):
    enet_gt_path = enet_patient_dir / 'GT.nii.gz'
    nnunet_gt_path = nnunet_patient_dir / 'GT.nii.gz'

    print(f"Copying {new_gt_path} to {enet_gt_path} and {nnunet_gt_path}...")
    shutil.copy(new_gt_path, enet_gt_path)
    shutil.copy(new_gt_path, nnunet_gt_path)
    print(f"Copied {new_gt_path} to ENet and nnU-Net folders.")

# Main script execution
patient_dirs = sorted(Path('../data/segthor_train/train').glob('Patient_*'))

# Check if patient directories are found and print the absolute path
print(f"Looking for patients in: {Path('../data/segthor_train/train').absolute()}")

if not patient_dirs:
    print(f"No patients found in {Path('data/data/segthor_train/train').absolute()}. Check the folder structure.")
else:
    print(f"Found {len(patient_dirs)} patients.")

for patient_dir in patient_dirs:
    patient_number = patient_dir.name.split('_')[1]
    print(f"Processing Patient {patient_number}...")

    gt_path = patient_dir / 'GT.nii.gz'
    moving_image_path = patient_dir / f'Patient_{patient_number}.nii.gz'

    # Check if both GT and moving image exist
    if not gt_path.exists() or not moving_image_path.exists():
        print(f"Files for Patient {patient_number} are missing. Skipping.")
        continue

    # Create directories for ENet and nnU-Net if they don't exist
    enet_patient_dir, nnunet_patient_dir = create_patient_directories(patient_number)

    try:
        print(f"Loading images for Patient {patient_number}...")
        fixed_image = (nib.load(gt_path).get_fdata() == replace_index).astype(np.float32)
        moving_image = (nib.load(moving_image_path).get_fdata() == replace_index).astype(np.float32)
    except Exception as e:
        print(f"Error loading data for Patient {patient_number}. Error: {e}")
        continue

    # Apply the saved affine transformation to the moving image
    print(f"Applying transformation to Patient {patient_number}...")
    transformed_image_data = apply_affine_transform(moving_image, 'transform.tfm')
    if transformed_image_data is None:
        print(f"Affine transformation failed for Patient {patient_number}. Skipping.")
        continue

    # Rename original GT to GT_old
    output_gt_old_path = patient_dir / 'GT_old.nii.gz'
    gt_path.rename(output_gt_old_path)

    # Save the new GT after applying the transformation to the fixed image (use the GT_old as reference)
    new_gt_path = patient_dir / 'GT.nii.gz'
    save_image_with_nibabel(transformed_image_data.round(), output_gt_old_path, new_gt_path, replace_index)

    # Save the transformed GT.nii.gz in ENet and nnU-Net directories
    copy_gt_to_enet_nnunet(new_gt_path, enet_patient_dir, nnunet_patient_dir)

    # Save the transformed images in ENet and nnU-Net directories (Patient_{number}.nii.gz)
    enet_patient_file = enet_patient_dir / f'Patient_{patient_number}.nii.gz'
    nnunet_patient_file = nnunet_patient_dir / f'Patient_{patient_number}.nii.gz'

    save_image_with_nibabel(transformed_image_data.round(), moving_image_path, enet_patient_file, replace_index)
    save_image_with_nibabel(transformed_image_data.round(), moving_image_path, nnunet_patient_file, replace_index)

    print(f"Processed Patient {patient_number}: GT and transformed data saved.")