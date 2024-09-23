import os
import torchio as tio

# Define patient folder base directory
base_folder = 'AI4MI/model_UNet/data/segthor_train/train'

# List of patient folders (01 to 40)
patient_folders = [f'Patient_{i:02d}' for i in range(1, 41)]

# Define augmentation setups for different scenarios
affine_transform = tio.RandomAffine(scales=(0.9, 1.1), degrees=10)  # Mild affine transformations
elastic_transform = tio.RandomElasticDeformation()  # Elastic deformations for soft tissue
gamma_transform = tio.RandomGamma(log_gamma=(0.7, 1.5))  # Contrast adjustment
noise_transform = tio.RandomNoise(mean=0, std=0.05)  # Moderate Gaussian noise

# Loop through each patient folder
for patient in patient_folders:
    gt_path = os.path.join(base_folder, patient, 'GT.nii.gz')  # Use the correct GT file
    
    if os.path.exists(gt_path):
        # Load the GT image (segmentations)
        image = tio.LabelMap(gt_path)
        
        # Apply affine + elastic transformations
        affine_augmented = affine_transform(image)
        elastic_augmented = elastic_transform(image)
        gamma_augmented = gamma_transform(image)
        noise_augmented = noise_transform(image)

        # Save augmentations
        affine_output_path = os.path.join(base_folder, patient, 'augmented_GT_affine.nii.gz')
        elastic_output_path = os.path.join(base_folder, patient, 'augmented_GT_elastic.nii.gz')
        gamma_output_path = os.path.join(base_folder, patient, 'augmented_GT_gamma.nii.gz')
        noise_output_path = os.path.join(base_folder, patient, 'augmented_GT_noise.nii.gz')
        
        affine_augmented.save(affine_output_path)
        elastic_augmented.save(elastic_output_path)
        gamma_augmented.save(gamma_output_path)
        noise_augmented.save(noise_output_path)
        
        print(f"Augmented images saved for {patient}")
        
    else:
        print(f"GT file not found for {patient}")