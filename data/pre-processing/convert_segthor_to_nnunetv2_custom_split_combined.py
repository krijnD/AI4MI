import argparse
import os
import shutil
import random
from collections import OrderedDict, defaultdict
from batchgenerators.utilities.file_and_folder_operations import subfolders, maybe_mkdir_p, save_json, join, isdir


"""
This scfript converts the SEGTHOR dataset (and or variations) into nnUNetv2 format with custom data splits.

How to use:
    python convert_segthor_to_nnunetv2_custom_split.py \
        --data_dir /path/to/segthor_dataset \
        --nnUNet_raw /path/to/nnUNet_raw_data \
        --nnUNet_preprocessed /path/to/nnUNet_preprocessed_data \
        --num_splits 5 \
        --num_val_cases 5 \
        --num_test_cases 5 \
        --dataset_id <unique_dataset_id> \
        --dataset_name <dataset_name>

Arguments:
    --data_dir: Path to the SEGTHOR dataset directory (e.g., segthor_train, segthor_affine).
    --nnUNet_raw: Path where the nnUNet raw data will be stored.
    --nnUNet_preprocessed: Path where the nnUNet preprocessed data will be stored.
    --num_splits: Number of splits for cross-validation (default is 5).
    --num_val_cases: Number of validation cases per split (default is 5).
    --num_test_cases: Number of test cases selected from training data (default is 5).
    --dataset_id: Unique dataset ID (e.g., 55, 56, 57) for nnUNetv2.
    --dataset_name: Name of the dataset variation (e.g., SegTHOR_Original, SegTHOR_Affine).

Description:
    - This script converts a SEGTHOR dataset variation into nnUNetv2 format.
    - It randomly selects test patients from the training data.
    - Custom splits for training and validation are created.
    - The converted data is saved under 'Dataset<dataset_id>_<dataset_name>' in the nnUNet raw data directory.

Steps:
    1. Convert the Dataset:
       Run the script to convert the SEGTHOR dataset variation into nnUNetv2 format.
    2. Run Preprocessing:
       Execute `nnUNetv2_plan_and_preprocess` with the specified dataset ID.
    3. Create Custom Splits:
       Run the script again to create custom splits (if not already created).
    4. Train the Model:
       Use `nnUNetv2_train` with the dataset ID to train the model.

"""

def create_splits(patient_id_mapping, remaining_patient_names, num_val_cases):
    """
    Creates custom splits for validation, assigning num_val_cases base patients to the validation set.
    Only the 'corrected_GT' augmentations of these base patients are assigned to the validation set.
    All other cases remain in the training set.
    """
    base_patient_ids = list(patient_id_mapping.keys())
    random.shuffle(base_patient_ids)
    
    # Select num_val_cases base patients for validation
    val_base_ids = base_patient_ids[:num_val_cases]
    
    val_cases = []
    train_cases = []
    
    for base_id in base_patient_ids:
        augmented_ids = patient_id_mapping[base_id]
        for aug_id in augmented_ids:
            if aug_id in remaining_patient_names:
                if base_id in val_base_ids and 'corrected_GT' in aug_id:
                    val_cases.append(aug_id)
                else:
                    train_cases.append(aug_id)
    
    splits = [{'train': train_cases, 'val': val_cases}]
    
    print(f"Total training cases: {len(train_cases)}")
    print(f"Total validation cases: {len(val_cases)}")
    
    return splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert multiple SEGTHOR augmentations to nnUNetv2 format with custom splits.")
    parser.add_argument('--data_dirs', type=str, nargs='+', required=True, help="List of dataset directory names (e.g., segthor_train, segthor_affine, segthor_elastic, segthor_noise).")
    parser.add_argument('--augmentations', type=str, nargs='+', required=True, help="List of augmentation names corresponding to data_dirs (e.g., corrected_GT, affine, elastic, noise).")
    parser.add_argument('--nnUNet_raw', type=str, required=True, help="Path to nnUNet raw data directory.")
    parser.add_argument('--nnUNet_preprocessed', type=str, required=True, help="Path to nnUNet preprocessed data directory.")
    parser.add_argument('--num_splits', type=int, default=1, help="Number of splits for cross-validation.")
    parser.add_argument('--num_val_cases', type=int, default=5, help="Number of validation cases per split.")
    parser.add_argument('--num_test_cases', type=int, default=5, help="Number of test cases.")
    parser.add_argument('--dataset_id', type=int, required=True, help="Unique dataset ID.")
    parser.add_argument('--dataset_name', type=str, required=True, help="Dataset name (e.g., SegTHOR_Combined).")
    args = parser.parse_args()

    # Base directory where all data_dirs are located
    base = '/home/kdignumsepu/AI4MI/data' 
    nnUNet_raw = args.nnUNet_raw
    nnUNet_preprocessed = args.nnUNet_preprocessed

    task_id = args.dataset_id
    task_name = args.dataset_name

    # nnUNetv2 expects datasets to be named as DatasetXXX_Name
    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    all_patient_names = []
    patient_id_mapping = defaultdict(list)  # Mapping from base patient ID to augmented patient IDs

    # Iterate over each dataset directory and corresponding augmentation
    for data_dir, augmentation in zip(args.data_dirs, args.augmentations):
        data_dir_path = join(base, data_dir)
        patient_names = subfolders(join(data_dir_path, "train"), join=False)
        patient_names.sort()
        
        for p in patient_names:
            augmented_patient_id = f"{p}_{augmentation}"
            curr = join(data_dir_path, "train", p)
            label_file = join(curr, "GT.nii.gz")
            image_file = join(curr, f"{p}.nii.gz")
            
            # Copy images and labels with new augmented_patient_id
            shutil.copy(image_file, join(imagestr, augmented_patient_id + "_0000.nii.gz"))
            shutil.copy(label_file, join(labelstr, augmented_patient_id + ".nii.gz"))
            
            all_patient_names.append(augmented_patient_id)
            
            # Map base patient ID to augmented IDs
            patient_id_mapping[p].append(augmented_patient_id)

    # Randomly select test patients from base patient IDs (segthor_train)
    base_patient_ids = list(patient_id_mapping.keys())
    random.seed(1234)  # For reproducibility
    test_base_patient_ids = random.sample(base_patient_ids, args.num_test_cases)

    # Collect only the 'corrected_GT' augmented patient IDs for test set
    test_patient_names = []
    for base_id in test_base_patient_ids:
        # Find the 'corrected_GT' augmented ID
        corrected_GT_id = next((aug_id for aug_id in patient_id_mapping[base_id] if 'corrected_GT' in aug_id), None)
        if corrected_GT_id:
            test_patient_names.append(corrected_GT_id)

    # Remaining patient names for training and validation
    remaining_patient_names = [p for p in all_patient_names if p not in test_patient_names]

    print(f"Total cases before split: {len(all_patient_names)}")
    print(f"Test cases: {len(test_patient_names)}")
    print(f"Remaining cases for training/validation: {len(remaining_patient_names)}")

    # Move test data to imagesTs and labelsTs
    for p in test_patient_names:
        # Move images
        src_image = join(imagestr, p + "_0000.nii.gz")
        dst_image = join(imagests, p + "_0000.nii.gz")
        shutil.move(src_image, dst_image)
        
        # Move labels
        src_label = join(labelstr, p + ".nii.gz")
        dst_label = join(labelsts, p + ".nii.gz")
        shutil.move(src_label, dst_label)

    # Create the dataset JSON for nnUNetv2
    dataset_json = OrderedDict()

    # nnUNetv2 requires 'channel_names', 'labels', 'file_ending', and 'numTraining'
    dataset_json['channel_names'] = {
        "0": "CT",
    }

    # Labels are specified with names as keys and integer IDs as values
    dataset_json['labels'] = {
        "background": 0,
        "esophagus": 1,
        "heart": 2,
        "trachea": 3,
        "aorta": 4,
    }

    # Specify the number of training cases
    dataset_json['numTraining'] = len(remaining_patient_names)

    # Specify the file ending
    dataset_json['file_ending'] = ".nii.gz"

    # Save the dataset JSON
    save_json(dataset_json, os.path.join(out_base, "dataset.json"), sort_keys=False)

    print(f"{task_name} dataset has been converted to nnUNetv2 format at {out_base}")

    # Ensure preprocessing is done before creating splits
    preprocessed_folder = join(nnUNet_preprocessed, foldername)
    if not isdir(preprocessed_folder):
        print(f"Preprocessed data not found at {preprocessed_folder}. Please run nnUNetv2_plan_and_preprocess first.")
    else:
        # Create custom splits
        splits = create_splits(patient_id_mapping, remaining_patient_names, args.num_splits, args.num_val_cases)

        # Path to the splits file in the preprocessed data directory
        splits_file = join(preprocessed_folder, 'splits_final.json')

        # Save the splits file
        save_json(splits, splits_file)

        print(f"Custom splits have been created and saved to {splits_file}")