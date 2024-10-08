import os
import shutil
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform  # To apply affine matrix
import torchio as tio
import hashlib

# --- Step 1: Move segthor_train and delete data/data ---
def move_segthor_train():
    base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Current path and new path for segthor_train
    current_segthor_train_path = os.path.join(base_data_dir, 'data', 'segthor_train')
    new_segthor_train_path = os.path.join(base_data_dir, 'segthor_train')
    
    if os.path.exists(current_segthor_train_path):
        if not os.path.exists(new_segthor_train_path):
            shutil.move(current_segthor_train_path, new_segthor_train_path)
            print(f"Moved segthor_train from {current_segthor_train_path} to {new_segthor_train_path}")
        else:
            print(f"Target segthor_train folder already exists: {new_segthor_train_path}")
    else:
        print(f"No segthor_train folder found in {current_segthor_train_path}")
    
    # Remove the now-empty data/data folder
    data_data_path = os.path.join(base_data_dir, 'data')
    if os.path.exists(data_data_path) and os.path.isdir(data_data_path):
        try:
            os.rmdir(data_data_path)
            print(f"Removed empty folder: {data_data_path}")
        except OSError:
            print(f"Could not remove {data_data_path} - folder not empty")

# --- Step 2: Create augmentation folders ---
def create_augmentation_folders():
    base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Paths for the new augmentation folders (segthor_affine, segthor_elastic, segthor_noise)
    new_folders = ['segthor_affine', 'segthor_elastic', 'segthor_noise']
    
    for folder in new_folders:
        folder_path = os.path.join(base_data_dir, folder, 'train')
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder and train subfolder: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")

# --- Helper Function: Hashing File Content ---
def get_file_hash(file_path):
    """Get the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# --- Step 3: Transform GT.nii.gz files ---
def compute_dice_coefficient(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    volume_sum = mask1.sum() + mask2.sum()
    if volume_sum == 0:
        return 1.0  # Both masks are empty
    return 2.0 * intersection / volume_sum

def transform_gt_files():
    base_folder = '../segthor_train/train'
    
    TR = np.asarray([[1, 0, 0, 50],
                     [0,  1, 0, 40],  
                     [0, 0, 1, 15],  
                     [0, 0, 0, 1]])

    DEG: int = 27
    ϕ: float = - DEG / 180 * np.pi
    RO = np.asarray([[np.cos(ϕ), -np.sin(ϕ), 0, 0],  
                     [np.sin(ϕ),  np.cos(ϕ), 0, 0],  
                     [     0,         0,     1, 0],  
                     [     0,         0,     0, 1]])

    X_bar: float = 275
    Y_bar: float = 200
    Z_bar: float = 0
    C1 = np.asarray([[1, 0, 0, X_bar],
                     [0, 1, 0, Y_bar],
                     [0, 0, 1, Z_bar],
                     [0, 0, 0,    1]])
    C2 = np.linalg.inv(C1)

    AFF = C1 @ RO @ C2 @ TR
    INV = np.linalg.inv(AFF)
    
    patient_folders = [f'Patient_{i:02d}' for i in range(1, 41)]
    
    for patient in patient_folders:
        gt_path = os.path.join(base_folder, patient, 'GT.nii.gz')
        gt_old_path = os.path.join(base_folder, patient, 'GT_old.nii.gz')  # Backup for old GT

        # Check if GT has already been transformed
        if os.path.exists(gt_old_path):
            print(f"Transformation already done for {patient}, skipping...")
            continue
        
        if os.path.exists(gt_path):
            print(f"Processing {gt_path} for transformation...")
            
            # Rename the old GT file before transformation
            os.rename(gt_path, gt_old_path)
            
            img = nib.load(gt_old_path)
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
            nib.save(aligned_img, gt_path)
            print(f"Saved transformed GT for {patient} at {gt_path}")

            if patient == 'Patient_27':
                good_gt_path = os.path.join(base_folder, 'Patient_27', 'GT2.nii.gz')
                good_img = nib.load(good_gt_path)
                good_gt = good_img.get_fdata()
                good_heart_segmentation = (good_gt == 2).astype(np.uint8)

                dice = compute_dice_coefficient(shifted_heart, good_heart_segmentation)
                print(f"Dice Coefficient for Patient 27 (Corrected GT vs GT2): {dice:.4f}")
        else:
            print(f"GT file not found for {patient}")

# --- Step 4: Apply augmentations and save in correct folders ---
def apply_augmentations():
    base_folder = '../segthor_train/train'
    
    output_folders = {
        'affine': '../segthor_affine/train',
        'elastic': '../segthor_elastic/train',
        'noise': '../segthor_noise/train',
    }

    patient_folders = [f'Patient_{i:02d}' for i in range(1, 41)]

    affine_transform = tio.RandomAffine(scales=(0.9, 1.1), degrees=10)
    elastic_transform = tio.RandomElasticDeformation()
    noise_transform = tio.RandomNoise(mean=0, std=0.05)

    for patient in patient_folders:
        ct_path = os.path.join(base_folder, patient, f'{patient}.nii.gz')
        gt_path = os.path.join(base_folder, patient, 'GT.nii.gz')
        augmented_affine_ct_path = os.path.join(output_folders['affine'], patient, f'{patient}.nii.gz')

        if os.path.exists(augmented_affine_ct_path):  # Check if augmented files exist
            print(f"Augmentations already done for {patient}, skipping...")
            continue

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
    move_segthor_train()  # Step 1: Move and clean up folders
    create_augmentation_folders()  # Step 2: Create new augmentation folders
    transform_gt_files()  # Step 3: Transform GT.nii.gz files before augmentations
    apply_augmentations()  # Step 4: Apply augmentations and save to correct folders