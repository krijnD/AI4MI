import os
import shutil
import torchio as tio

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

# --- Step 3: Apply augmentations and save in correct folders ---
def apply_augmentations():
    # Define the base folder for patient data
    base_folder = '../segthor_train/train'  # Now one level up after moving

    # Define paths for saving augmented files
    output_folders = {
        'affine': '../segthor_affine/train',
        'elastic': '../segthor_elastic/train',
        'noise': '../segthor_noise/train',
    }

    patient_folders = [f'Patient_{i:02d}' for i in range(1, 41)]

    # Define augmentation setups
    affine_transform = tio.RandomAffine(scales=(0.9, 1.1), degrees=10)
    elastic_transform = tio.RandomElasticDeformation()
    noise_transform = tio.RandomNoise(mean=0, std=0.05)

    for patient in patient_folders:
        ct_path = os.path.join(base_folder, patient, f'{patient}.nii.gz')
        gt_path = os.path.join(base_folder, patient, 'GT.nii.gz')

        if os.path.exists(ct_path) and os.path.exists(gt_path):
            ct_image = tio.ScalarImage(ct_path)
            gt_image = tio.LabelMap(gt_path)
            subject = tio.Subject(ct=ct_image, gt=gt_image)

            # Apply augmentations and save in the correct folders
            for aug_type, transform, folder_key in zip(
                    ['affine', 'elastic', 'noise'],
                    [affine_transform, elastic_transform, noise_transform],
                    ['affine', 'elastic', 'noise']):

                augmented = transform(subject)
                aug_patient_folder = os.path.join(output_folders[folder_key], patient)
                os.makedirs(aug_patient_folder, exist_ok=True)

                # Save augmented files
                augmented['ct'].save(os.path.join(aug_patient_folder, f'{patient}.nii.gz'))
                augmented['gt'].save(os.path.join(aug_patient_folder, 'GT.nii.gz'))

            print(f"Augmentations saved for {patient} in affine, elastic, and noise folders.")
        else:
            print(f"CT or GT file not found for {patient}")

# --- Main execution ---
if __name__ == "__main__":
    move_segthor_train()
    create_augmentation_folders()
    apply_augmentations()