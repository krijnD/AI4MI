import os
import shutil
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
import torchio as tio
import torch  # Import torch for tensor operations


# Function to compute the Dice Coefficient
def compute_dice_coefficient(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    volume_sum = mask1.sum() + mask2.sum()
    if volume_sum == 0:
        return 1.0  # Both masks are empty
    return 2.0 * intersection / volume_sum

# --- Step 1: Prepare segthor_original or segthor_original_dummy ---
def prepare_segthor_original(downloaded_data_dir, segthor_original_dir):
    """Copies the downloaded data into segthor_original (or _dummy) and ensures the train subfolder structure."""
    train_dir = os.path.join(segthor_original_dir, 'train')
    if not os.path.exists(segthor_original_dir):
        shutil.copytree(downloaded_data_dir, train_dir)
        print(f"Created {segthor_original_dir} from {downloaded_data_dir}")
        
        # After copying, delete the original 'data/segthor_train' folder two levels up
        segthor_train_dir_to_delete = os.path.abspath(os.path.join(downloaded_data_dir, '..', '..'))
        if os.path.exists(segthor_train_dir_to_delete):
            shutil.rmtree(segthor_train_dir_to_delete)
            print(f"Deleted original folder: {segthor_train_dir_to_delete}")
    else:
        print(f"{segthor_original_dir} already exists. Skipping this step.")

# --- Step 2: Create augmentation folders with _dummy suffix if needed ---
def create_augmentation_folders(base_data_dir, dummy=False):
    """Creates augmentation folders with train subfolders and appends _dummy if needed."""
    suffix = '_dummy' if dummy else ''
    new_folders = [f'segthor_affine{suffix}', f'segthor_elastic{suffix}', f'segthor_noise{suffix}']
    
    for folder in new_folders:
        folder_path = os.path.join(base_data_dir, folder, 'train')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder and train subfolder: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")

# --- Step 3: Transform GT.nii.gz files ---
def transform_gt_files(segthor_original_dir, segthor_train_dir):
    """Transforms the GT.nii.gz files from segthor_original and stores them in segthor_train."""
    train_dir = os.path.join(segthor_train_dir, 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        print(f"Created {segthor_train_dir} directory.")

    patient_folders = [f'Patient_{i:02d}' for i in range(1, 41)]
    
    # Transformation parameters
    TR = np.asarray([[1, 0, 0, 50],
                     [0,  1, 0, 40],  
                     [0, 0, 1, 15],  
                     [0, 0, 0, 1]])
    DEG = 27
    ϕ = - DEG / 180 * np.pi
    RO = np.asarray([[np.cos(ϕ), -np.sin(ϕ), 0, 0],  
                     [np.sin(ϕ),  np.cos(ϕ), 0, 0],  
                     [     0,         0,     1, 0],  
                     [     0,         0,     0, 1]])

    X_bar, Y_bar, Z_bar = 275, 200, 0
    C1 = np.asarray([[1, 0, 0, X_bar],
                     [0, 1, 0, Y_bar],
                     [0, 0, 1, Z_bar],
                     [0, 0, 0,    1]])
    C2 = np.linalg.inv(C1)

    AFF = C1 @ RO @ C2 @ TR
    INV = np.linalg.inv(AFF)

    # Process each patient's GT and CT files
    for patient in patient_folders:
        # Check if CT file exists directly in the Patient folder (no 'train' subfolder)
        ct_path = os.path.join(segthor_original_dir, patient, f'{patient}.nii.gz')
        
        # Fallback to 'train' folder if CT file is not found
        if not os.path.exists(ct_path):
            ct_path = os.path.join(segthor_original_dir, 'train', patient, f'{patient}.nii.gz')
        
        new_gt_path = os.path.join(segthor_train_dir, 'train', patient, 'GT.nii.gz')
        
        # Ensure patient folder in segthor_train exists
        new_patient_folder = os.path.join(segthor_train_dir, 'train', patient)
        os.makedirs(new_patient_folder, exist_ok=True)

        # Copy the CT scan (Patient_{number}.nii.gz) to segthor_train if it exists
        if os.path.exists(ct_path):
            shutil.copy(ct_path, os.path.join(new_patient_folder, f'{patient}.nii.gz'))
            print(f"Copied {ct_path} to {new_patient_folder}")
        else:
            print(f"CT file not found for {patient}")

        # Apply transformation to the GT file and save in segthor_train
        original_gt_path = os.path.join(segthor_original_dir, patient, 'GT.nii.gz')
        if not os.path.exists(original_gt_path):
            original_gt_path = os.path.join(segthor_original_dir, 'train', patient, 'GT.nii.gz')
        
        if os.path.exists(original_gt_path):
            print(f"Processing {original_gt_path} for transformation...")

            img = nib.load(original_gt_path)
            gt = img.get_fdata()
            original_affine = img.affine

            # Check the unique values and range before conversion
            unique_values = np.unique(gt)
            print(f"GT unique values for {patient}: {unique_values}")
            if not np.all(np.isin(unique_values, [0, 1, 2, 3, 4])):  # Adjust range if you expect more classes
                raise ValueError(f"Unexpected values found in the GT for {patient}: {unique_values}")

            # Convert ground truth to uint8 if necessary
            if gt.dtype != np.uint8:
                print(f"Converting GT for {patient} from {gt.dtype} to uint8")
                gt = np.round(gt).astype(np.uint8)

            heart_segmentation = (gt == 2).astype(np.uint8)
            shifted_heart = affine_transform(heart_segmentation, INV[:3, :3], offset=INV[:3, 3])
            shifted_heart = np.round(shifted_heart).astype(np.uint8)

            transformed_data = np.copy(gt)
            transformed_data[gt == 2] = 0  # Remove original heart
            transformed_data[shifted_heart == 1] = 2  # Replace with transformed heart

            transformed_data = transformed_data.astype(np.uint8)  # Ensure GT remains uint8
            aligned_img = nib.Nifti1Image(transformed_data, affine=original_affine)
            nib.save(aligned_img, new_gt_path)
            print(f"Saved transformed GT for {patient} at {new_gt_path}")
        else:
            print(f"GT file not found for {patient}")


# --- Step 4: Apply augmentations and save in folders with _dummy suffix if needed ---
def apply_augmentations(segthor_train_dir, base_data_dir, dummy=False):
    """Applies augmentations to the transformed data in segthor_train and appends _dummy if needed."""
    suffix = '_dummy' if dummy else ''
    output_folders = {
        'affine': os.path.join(base_data_dir, f'segthor_affine{suffix}', 'train'),
        'elastic': os.path.join(base_data_dir, f'segthor_elastic{suffix}', 'train'),
        'noise': os.path.join(base_data_dir, f'segthor_noise{suffix}', 'train')
    }

    patient_folders = [f'Patient_{i:02d}' for i in range(1, 41)]

    affine_transform = tio.RandomAffine(scales=(0.9, 1.1), degrees=10)
    elastic_transform = tio.RandomElasticDeformation(num_control_points=5, max_displacement=(20, 20, 20))
    noise_transform = tio.RandomNoise(mean=0, std=0.1)

    for patient in patient_folders:
        ct_path = os.path.join(segthor_train_dir, 'train', patient, f'{patient}.nii.gz')
        gt_path = os.path.join(segthor_train_dir, 'train', patient, 'GT.nii.gz')

        if os.path.exists(ct_path) and os.path.exists(gt_path):
            ct_image = tio.ScalarImage(ct_path)
            gt_image = tio.LabelMap(gt_path)
            subject = tio.Subject(ct=ct_image, gt=gt_image)

            # Convert the label map (GT) to uint8 using PyTorch's .to() method
            gt_image.set_data(gt_image.data.to(torch.uint8))

            for aug_type, transform, folder_key in zip(
                    ['affine', 'elastic', 'noise'],
                    [affine_transform, elastic_transform, noise_transform],
                    ['affine', 'elastic', 'noise']):

                augmented = transform(subject)

                # Convert the augmented GT back to uint8
                augmented['gt'].set_data(augmented['gt'].data.to(torch.uint8))

                # Check that the augmented CT and GT data have the correct types
                assert augmented['ct'].data.dtype in [torch.float32, torch.int16], f"CT dtype mismatch: {augmented['ct'].data.dtype}"
                assert augmented['gt'].data.dtype == torch.uint8, f"GT dtype mismatch: {augmented['gt'].data.dtype}"

                aug_patient_folder = os.path.join(output_folders[folder_key], patient)
                os.makedirs(aug_patient_folder, exist_ok=True)

                augmented['ct'].save(os.path.join(aug_patient_folder, f'{patient}.nii.gz'))
                augmented['gt'].save(os.path.join(aug_patient_folder, 'GT.nii.gz'))

            print(f"Augmentations saved for {patient} in affine, elastic, and noise folders.")
        else:
            print(f"CT or GT file not found for {patient}")



# --- Main execution ---
if __name__ == "__main__":
    # Set to True if working with the dummy dataset
    dummy = True

    # Define the appropriate dataset based on dummy flag
    base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    suffix = '_dummy' if dummy else ''
    
    downloaded_data_dir = os.path.join(base_data_dir, f'data/segthor_train{suffix}', 'train')  # Original downloaded data (or dummy)
    segthor_original_dir = os.path.join(base_data_dir, f'segthor_original{suffix}')  # Directory for original, untransformed data
    segthor_train_dir = os.path.join(base_data_dir, f'segthor_train{suffix}')  # Directory for transformed, corrected GTs

    # Step 1: Prepare segthor_original (or _dummy)
    prepare_segthor_original(downloaded_data_dir, segthor_original_dir)

    # Step 2: Transform GT files and create segthor_train (or _dummy)
    transform_gt_files(segthor_original_dir, segthor_train_dir)

    # Step 3: Create augmentation folders (with _dummy if needed)
    create_augmentation_folders(base_data_dir, dummy=dummy)

    # Step 4: Apply augmentations from segthor_train (or _dummy)
    apply_augmentations(segthor_train_dir, base_data_dir, dummy=dummy)
