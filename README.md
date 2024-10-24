# AI4MI
<!-- ![Pipeline](AI4MI/pipeline.png) -->
<img src="AI4MI/pipeline.png" alt="pipeline" width="100"/>
## Data Preparation and Augmentation
If you have the SEGTHOR dataset in a zip file, follow these steps to prepare and augment the data:
1.	Unzip the Data: Extract data.zip into the data/ folder.
2.	Ensure Required Files: Make sure the transform.tfm file is located in the same directory as the transform.py script (AI4MI/data/pre-processing/).
3.	Run the Transformation Script: The transform.py script creates several data augmentations (affine, elastic, original, train and noise) to expand the dataset. Execute the script as follows:
```bash
python AI4MI/data/pre-processing/transform.py
```
This script processes the dataset, generates the transformations, and places the augmented data into respective folders like segthor_affine, segthor_elastic, and segthor_noise.

## Enet Installation and Usage
To initialize the submodules run
```bash
cd AI4MI
git submodule init
git submodule update
```

Ensure you are using Python 3.10 or later. You can create a virtual environment and install the required packages from requirements.txt:

```bash
python -m venv ai4mi
source ai4mi/bin/activate
python -m pip install -r requirements.txt
```
## Enet Data Preprocessing
After having ran the transformation.py script, execute the following make commands slice the data and make it Enet compatible

```bash
make data/slice_segthor_train
make data/slice_segthor_affine
make data/slice_segthor_elastic
make data/slice_segthor_noise
```

## nnUNet Installation and Usage

### Installation
This section provides instructions on how to install and use nnUNet for training, inference, evaluation, and data conversion.

First, clone the AI4MI repository and install nnUNet:

```bash
git clone https://github.com/krijnD/AI4MI.git
cd AI4MI/nnUNet
pip install -e .
```
### Data Conversion and Preprocessing
Before converting data, ensure the dataset follows the correct folder structure. This repository is a clone of the original nnUNet repository, and for detailed instructions on creating the correct structure, please refer to the original [nnUNet repository](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).


Before using nnUNet, the SEGTHOR dataset must be converted to the required format. You can use the following script:
```bash
python data/pre-processing/convert_segthor_to_nnunetv2_custom_split.py \
  --data_dir <data_dir> \
  --nnUNet_raw <nnUNet_raw_data_dir> \
  --nnUNet_preprocessed <nnUNet_preprocessed_dir> \
  --num_splits <number_of_splits> \
  --num_val_cases <num_validation_cases> \
  --num_test_cases <num_test_cases> \
  --dataset_id <dataset_id> \
  --dataset_name <dataset_name>
```
Once the conversion is complete, preprocess the data to prepare it for training:
```bash
nnUNetv2_plan_and_preprocess -d <dataset_id > --verify_dataset_integrity
```

### Training with nnUNet
To train a model using nnUNet, use the nnUNetv2_train command with the appropriate parameters for your dataset, configuration, and trainer class.

```bash
nnUNetv2_train <dataset_id> <configuration> <fold> -tr <trainer_class>
```

Examples of training with different loss functions:
```bash
nnUNetv2_train 55 2d 0 -tr nnUNetTrainerDiceCELoss_noSmooth
nnUNetv2_train 55 2d 0 -tr nnUNetTrainerFocalLoss
nnUNetv2_train 55 2d 0 -tr nnUNetTrainerTverskyLoss
```

### Inference
After training, perform inference using the nnUNetv2_predict command.
```bash
nnUNetv2_predict -i <input_folder> -o <output_folder> -d <dataset_id> -c <configuration> -tr <trainer_class> -f <fold>
```

Example of Inference with the Dice + CE loss trained model:
```bash
nnUNetv2_predict -i <imagesTs_dir> \
  -o <predictions_dir> \
  -d 55 \
  -c 2d \
  -tr nnUNetTrainerDiceCELoss_noSmooth \
  -f 0
```

### Post-processing with CRF

To apply Conditional Random Fields (CRF) as post-processing:
1. Perform inference with probabilities saved:
```bash
nnUNetv2_predict -i <imagesTs_dir> \
  -o <predictions_with_probs_dir> \
  -d <dataset_id> \
  -c <model_configuration> \
  -tr nnUNetTrainerDiceCELoss_noSmooth \
  -f <cross_validation_folds> \
  --save_probabilities
```
2. Apply CRF:
```bash
python nnUNet/results/CRF_experiments/apply_crf_nnUNetV2.py \
--image_dir <path_to_imagesTs_dir> \
  --prediction_dir <path_to_predictions_with_probs_dir> \
  --output_dir <path_to_crf_output_dir> \
  --gt_dir <path_to_labelsTs_dir> \
  --num_classes <num_classes> \
  --t <t_value> \
  --sxy_gaussian <sxy_gaussian_value> \
  --compat_gaussian <compat_gaussian_value> \
  --sxy_bilateral <sxy_bilateral_value> \
  --srgb_bilateral <srgb_bilateral_value> \
  --compat_bilateral <compat_bilateral_value>
```