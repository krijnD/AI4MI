import os
import torchio as tio

# Define patient folder base directory relative to the current script location
base_folder = '../data/segthor_train/train'

# List of patient folders (01 to 40)
patient_folders = [f'Patient_{i:02d}' for i in range(1, 41)]

# Define augmentation setups for different scenarios
affine_transform = tio.RandomAffine(scales=(0.9, 1.1), degrees=10)  # Mild affine transformations
elastic_transform = tio.RandomElasticDeformation()  # Elastic deformations for soft tissue
noise_transform = tio.RandomNoise(mean=0, std=0.05)  # Moderate Gaussian noise

# Loop through each patient folder
for patient in patient_folders:
    # Corrected paths to CT volume and GT segmentation files
    ct_path = os.path.join(base_folder, patient, f'{patient}.nii.gz')  # CT volume
    gt_path = os.path.join(base_folder, patient, 'GT.nii.gz')  # Ground truth segmentation
    
    if os.path.exists(ct_path) and os.path.exists(gt_path):
        # Load the CT and GT images
        ct_image = tio.ScalarImage(ct_path)
        gt_image = tio.LabelMap(gt_path)
        
        # Compose both images into one subject
        subject = tio.Subject(
            ct=ct_image,
            gt=gt_image
        )
        
        # Apply transformations to both the CT and the GT segmentation
        affine_augmented = affine_transform(subject)
        elastic_augmented = elastic_transform(subject)
        noise_augmented = noise_transform(subject)

        # Save augmented CT and GT segmentation files
        affine_output_ct = os.path.join(base_folder, patient, 'augmented_CT_affine.nii.gz')
        affine_output_gt = os.path.join(base_folder, patient, 'augmented_GT_affine.nii.gz')
        
        elastic_output_ct = os.path.join(base_folder, patient, 'augmented_CT_elastic.nii.gz')
        elastic_output_gt = os.path.join(base_folder, patient, 'augmented_GT_elastic.nii.gz')
        
        noise_output_ct = os.path.join(base_folder, patient, 'augmented_CT_noise.nii.gz')
        noise_output_gt = os.path.join(base_folder, patient, 'augmented_GT_noise.nii.gz')
        
        # Save CT and GT augmentations
        affine_augmented['ct'].save(affine_output_ct)
        affine_augmented['gt'].save(affine_output_gt)
        
        elastic_augmented['ct'].save(elastic_output_ct)
        elastic_augmented['gt'].save(elastic_output_gt)
        
        noise_augmented['ct'].save(noise_output_ct)
        noise_augmented['gt'].save(noise_output_gt)
        
        print(f"Augmented images saved for {patient}")
        
    else:
        print