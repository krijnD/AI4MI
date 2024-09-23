import torchio as tio

# Load your NIfTI files
patient_folder = 'AI4MI/model_UNet/data/segthor_train/train'
gt_path = f'{patient_folder}/GT.nii.gz'
image = tio.ScalarImage(gt_path)

# Define augmentation transformations
transform = tio.Compose([
    tio.RandomFlip(axes=(0, 1)),  # Random flipping along x and y
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10),  # Random scaling and rotation
    tio.RandomElasticDeformation(),  # Elastic deformation for realistic distortions
    tio.RandomGamma(log_gamma=(0.7, 1.5)),  # Adjust contrast
    tio.RandomNoise(mean=0, std=0.1),  # Adding random noise
])

# Apply augmentation
augmented_image = transform(image)

# Save augmented image
augmented_image.save(f'{patient_folder}/augmented_GT.nii.gz')