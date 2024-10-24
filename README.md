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
### Training
To train models with nnUNet, use the nnUNetv2_train command with the appropriate dataset ID, configuration, fold, and trainer class (for different loss functions).

```bash
nnUNetv2_train <dataset_id> <configuration> <fold> -tr <trainer_class>
```


