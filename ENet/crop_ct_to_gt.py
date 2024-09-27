import os
import nibabel as nib
import numpy as np

# Define paths to the data
base_folder = '../data/segthor_train/train'
patient_folders = [f'Patient_{i:02d}' for i in range(1, 41)]

for patient in patient_folders:
    # Paths to CT and GT files
    ct_path = os.path.join(base_folder, patient, f'{patient}.nii.gz')
    gt_path = os.path.join(base_folder, patient, 'GT.nii.gz')

    if os.path.exists(ct_path) and os.path.exists(gt_path):
        # Load CT and GT images using nibabel
        ct_img = nib.load(ct_path)
        gt_img = nib.load(gt_path)
        
        # Get the image data as numpy arrays
        ct_data = ct_img.get_fdata()
        gt_data = gt_img.get_fdata()
        
        # Find non-zero slices in the GT segmentation along the x, y, and z axes
        non_zero_x = np.where(np.any(gt_data, axis=(1, 2)))[0]
        non_zero_y = np.where(np.any(gt_data, axis=(0, 2)))[0]
        non_zero_z = np.where(np.any(gt_data, axis=(0, 1)))[0]
        
        # Get the bounding box for non-zero values
        x_min, x_max = non_zero_x[0], non_zero_x[-1]
        y_min, y_max = non_zero_y[0], non_zero_y[-1]
        z_min, z_max = non_zero_z[0], non_zero_z[-1]
        
        # Crop the CT scan to match the GT bounding box
        cropped_ct_data = ct_data[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
        
        # Create a new NIfTI image for the cropped CT scan
        cropped_ct_img = nib.Nifti1Image(cropped_ct_data, affine=ct_img.affine)
        
        # Save the cropped CT image
        cropped_ct_path = os.path.join(base_folder, patient, f'cropped_{patient}.nii.gz')
        nib.save(cropped_ct_img, cropped_ct_path)
        
        print(f'Cropped CT saved for {patient}: {cropped_ct_path}')
    else:
        print(f"CT or GT file not found for {patient}")