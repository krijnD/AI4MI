import os
import shutil
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform  # To apply affine matrix
import torchio as tio

# Function to compute the Dice Coefficient
def compute_dice_coefficient(mask1, mask2):
    """Computes the Dice Coefficient between two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    volume_sum = mask1.sum() + mask2.sum()
    if volume_sum == 0:
        return 1.0  # Both masks are empty
    return 2.0 * intersection / volume_sum

# --- Step 1: Prepare segthor_original ---
def prepare_segthor_original(downloaded_data_dir, segthor_original_dir):
    """Copies the downloaded data into segthor_original and ensures the train subfolder structure."""
    train_dir = os.path.join(segthor_original_dir, 'train')
    if not os.path.exists(segthor_original_dir):
        shutil.copytree(downloaded_data_dir, train_dir)  # Copy downloaded data to segthor_original/train
        print(f"Created segthor_original from {downloaded_data_dir}")
        
        # After copying, delete the original 'data/segthor_train' folder two levels up
        segthor_train_dir_to_delete = os.path.abspath(os.path.join(downloaded_data_dir, '..', '..'))
        if os.path.exists(segthor_train_dir_to_delete):
            shutil.rmtree(segthor_train_dir_to_delete)
            print(f"Deleted original folder: {segthor_train_dir_to_delete}")
    else:
        print(f"segthor_original already exists. Skipping this step.")

# --- Step 2: Create augmentation folders ---
def create_augmentation_folders(base_data_dir):
    """Creates augmentation folders with train subfolders."""
    new_folders = ['segthor_affine', 'segthor_elastic', 'segthor_noise']
    
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
        print(f"Created segthor_train directory.")

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
        original_gt_path = os.path.join(segthor_original_dir, 'train', patient, 'GT.nii.gz')
        new_gt_path = os.path.join(segthor_train_dir, 'train', patient, 'GT.nii.gz')
        ct_path = os.path.join(segthor_original_dir, 'train', patient, f'{patient}.nii.gz')

        # Ensure patient folder in segthor_train exists
        new_patient_folder = os.path.join(segthor_train_dir, 'train', patient)
        os.makedirs(new_patient_folder, exist_ok=True)

        # Copy the CT scan (Patient_{number}.nii.gz) to segthor_train
        if os.path.exists(ct_path):
            shutil.copy(ct_path, os.path.join(new_patient_folder, f'{patient}.nii.gz'))
            print(f"Copied {ct_path} to {new_patient_folder}")
        else:
            print(f"CT file not found for {patient}")

        # Apply transformation to the GT file and save in segthor_train
        if os.path.exists(original_gt_path):
            print(f"Processing {original_gt_path} for transformation...")

            img = nib.load(original_gt_path)
            gt = img.get_fdata()
            original_affine = img.affine

            heart_segmentation = (gt == 2).astype(np.uint8)
            shifted_heart = affine_transform(heart_segmentation, INV[:3, :3], offset=INV[:3, 3])
            shifted_heart = np.round(shifted_heart).astype(np.uint8)

            transformed_data = np.copy(gt)
            transformed_data[gt == 2] = 0  # Remove original heart
            transformed_data[shifted_heart == 1] = 2  # Replace with transformed heart

            transformed_data = transformed_data.astype(np.uint8)
            aligned_img = nib.Nifti1Image(transformed_data, affine=original_affine)
            nib.save(aligned_img, new_gt_path)
            print(f"Saved transformed GT for {patient} at {new_gt_path}")

            # If processing Patient_27, compare with GT2 and compute Dice Coefficient
            if patient == 'Patient_27':
                good_gt_path = os.path.join(segthor_original_dir, 'train', 'Patient_27', 'GT2.nii.gz')
                if os.path.exists(good_gt_path):
                    good_img = nib.load(good_gt_path)
                    good_gt = good_img.get_fdata()
                    good_heart_segmentation = (good_gt == 2).astype(np.uint8)
                    
                    # Compute Dice Coefficient
                    dice = compute_dice_coefficient(shifted_heart, good_heart_segmentation)
                    print(f"Dice Coefficient for Patient 27 (Corrected GT vs GT2): {dice:.4f}")
                else:
                    print(f"GT2 file not found for Patient 27.")

        else:
            print(f"GT file not found for {patient}")

# --- Step 4: Apply augmentations and save in correct folders ---
def apply_augmentations(segthor_train_dir, base_data_dir):
    """Applies augmentations to the transformed data in segthor_train."""
    output_folders = {
        'affine': os.path.join(base_data_dir, 'segthor_affine', 'train'),
        'elastic': os.path.join(base_data_dir, 'segthor_elastic', 'train'),
        'noise': os.path.join(base_data_dir, 'segthor_noise', 'train')
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

            for aug_type, transform, folder_key in zip(
                    ['affine', 'elastic', 'noise'],
                    [affine_transform, elastic_transform, noise_transform],
                    ['affine', 'elastic', 'noise']):

                augmented = transform(subject)
                aug_patient_folder = os.path.join(output_folders[folder_key], patient)
                os.makedirs(aug_patient_folder, exist_ok=True)

                augmented['ct'].save(os.path.join(aug_patient_folder, f'{patient}.nii.gz'))
                augmented['gt'].save(os.path.join(aug_patient_folder, 'GT.nii.gz'))

            print(f"Augmentations saved for {patient} in affine, elastic, and noise folders.")
        else:
            print(f"CT or GT file not found for {patient}")

# --- Main execution ---
if __name__ == "__main__":
    # Since we're running from 'AI4MI/data/pre-processing', we go up one level to access 'AI4MI/data'
    base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    downloaded_data_dir = os.path.join(base_data_dir, 'data', 'segthor_train', 'train')  # Original downloaded data
    segthor_original_dir = os.path.join(base_data_dir, 'segthor_original')  # Directory for original, untransformed data
    segthor_train_dir = os.path.join(base_data_dir, 'segthor_train')  # Directory for transformed, corrected GTs

    # Step 1: Prepare segthor_original (only if it doesn't exist)
    prepare_segthor_original(downloaded_data_dir, segthor_original_dir)

    # Step 2: Transform GT files and create segthor_train
    transform_gt_files(segthor_original_dir, segthor_train_dir)

    # Step 3: Create augmentation folders
    create_augmentation_folders(base_data_dir)

    # Step 4: Apply augmentations from segthor_train
    apply_augmentations(segthor_train_dir, base_data_dir)