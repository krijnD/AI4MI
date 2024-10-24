# AI4MI

## nnUNet Installation and Usage

### Installation
This section provides instructions on how to install and use nnUNet for training, inference, evaluation, and data conversion.

First, clone the AI4MI repository and install nnUNet:

```bash
git clone https://github.com/krijnD/AI4MI.git
cd AI4MI/nnUNet
pip install -e .
```
### Data Conversion
To convert the SEGTHOR dataset into the nnUNet format, use the provided  script. Example:
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
after converting the data it has to be preprocessed:
```bash
nnUNetv2_plan_and_preprocess -d <dataset_id > --verify_dataset_integrity
```

### Training with nnUNet
To train models with nnUNet, use the nnUNetv2_train command with the appropriate dataset ID, configuration, fold, and trainer class (for different loss functions).

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
To perform inference with a trained model, use the nnUNetv2_predict command.
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
  --image_dir <imagesTs_dir> \
  --prediction_dir <predictions_with_probs_dir> \
  --output_dir <crf_output_dir> \
  --gt_dir <labelsTs_dir> \
  --num_classes 5 \
  --t 5 \
  --sxy_gaussian 3 \
  --compat_gaussian 5 \
  --sxy_bilateral 80 \
  --srgb_bilateral 13 \
  --compat_bilateral 10
```