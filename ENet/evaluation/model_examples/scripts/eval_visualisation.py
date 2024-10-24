import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Define class names (assuming indices start from 1)
class_names = ['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta']

# Define custom palette
custom_palette = {
    'roze': '#eb8fd8',
    'groen': '#b9d4b4',
    'paars': '#ba94e9',
    'blue': '#4C8BE2',
    'orange': '#E27A3F',
    'grey_light': '#1F3240',
    'grey_dark': '#16242F'
}


# Define custom themes
def set_custom_dark_theme():
    sns.set_context('notebook', font_scale=1.2)
    sns.set_style({
        'axes.facecolor': custom_palette['grey_light'],
        'axes.edgecolor': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': 'white',
        'figure.facecolor': custom_palette['grey_light'],
        'text.color': 'white'
    })
    sns.set_palette([
        custom_palette['roze'],
        custom_palette['groen'],
        custom_palette['paars'],
        custom_palette['blue'],
        custom_palette['orange']
    ])


def set_custom_light_theme():
    sns.set_context('notebook', font_scale=1.2)
    sns.set_style({
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'grid.color': custom_palette["grey_light"],
        'figure.facecolor': 'white',
        'text.color': 'black'
    })
    sns.set_palette([
        custom_palette['roze'],
        custom_palette['groen'],
        custom_palette['paars'],
        custom_palette['blue'],
        custom_palette['orange']
    ])


# Define a palette for the models
model_palette = {
    'Elastic augmentation': custom_palette['roze'],
    'Noise augmentation': custom_palette['groen'],
    'Affine augmentation': custom_palette['paars'],
    'All augmentations': custom_palette['blue'],
    'Baseline': custom_palette['orange'],
    'CE Loss': custom_palette['roze'],
    'Dice + CE Loss': custom_palette['groen'],
    'Dice Loss': custom_palette['paars'],
    'Focal Loss': custom_palette['blue'],
    'Tversky Loss': custom_palette['orange'],
    'CRF': custom_palette['blue'],
    'Baseline': custom_palette['orange'],
    'Baseline 3d nnUNet': custom_palette['roze'],
    'Baseline 2d nnUNet': custom_palette['groen']
}

# network_dim = 2
# output_dir = 'nnUNet/results/vis_results/2d/loss'
# output_dir = 'nnUNet/results/vis_results/2d/augmentation'
# output_dir = 'nnUNet/results/vis_results/2d/CRF'

# Assume we have a dictionary of model names and corresponding json files
# Replace the paths with your actual json file paths

# 2D Results----------------------------------------------------------------------------------------------------------------
# json_files = {
#     'CE Loss': '/home/kdignumsepu/AI4MI/nnUNet/results/loss_experiments/corrected_Haussdorff/Corrected_GT_CE_loss_summary.json',
#     'Dice + CE Loss': '/home/kdignumsepu/AI4MI/nnUNet/results/loss_experiments/corrected_Haussdorff/Corrected_GT_Dice_CE_loss_summary.json',
#     'Dice Loss': '/home/kdignumsepu/AI4MI/nnUNet/results/loss_experiments/corrected_Haussdorff/Corrected_GT_Dice_loss_summary.json',
#     'Focal Loss': '/home/kdignumsepu/AI4MI/nnUNet/results/loss_experiments/corrected_Haussdorff/Corrected_GT_Focal_loss_summary.json'
#     # 'Tversky Loss': '/home/kdignumsepu/AI4MI/nnUNet/results/loss_experiments/corrected_Haussdorff/Corrected_GT_Tversky_loss_summary.json'

# }

# json_files = {
#     'Elastic augmentation': 'nnUNet/results/augmentation_results/Elastic_Dice_CE_loss_summary.json',
#     'Noise augmentation': 'nnUNet/results/augmentation_results/Noise_Dice_CE_loss_summary.json',
#     'Affine augmentation': 'nnUNet/results/augmentation_results/Affine_Dice_CE_loss_summary.json',
#     'All augmentations': 'nnUNet/results/augmentation_results/Combined_Dice_CE_loss_summary.json',
#     'Baseline': 'nnUNet/results/loss_experiments/corrected_Haussdorff/Corrected_GT_Dice_CE_loss_summary.json'
# }

# json_files = {
#     'CRF': 'nnUNet/results/CRF_experiments/CRF_output/2d/CRF_summary.json',
#     'Baseline': 'nnUNet/results/loss_experiments/corrected_Haussdorff/Corrected_GT_Dice_CE_loss_summary.json'
# }

# 3D Results----------------------------------------------------------------------------------------------------------------
network_dim = 3
# output_dir = 'nnUNet/results/vis_results/3d/loss'
# output_dir = 'nnUNet/results/vis_results/3d/augmentation'
output_dir = 'nnUNet/results/vis_results/3d/CRF'

# json_files = {
#     'CE Loss': 'nnUNet/results/3d_losses_experiments/3d_lowres_CE_loss_summary.json',
#     'Dice + CE Loss': 'nnUNet/results/3d_losses_experiments/3d_lowres_Dice_CE_loss_summary.json',
#     'Dice Loss': 'nnUNet/results/3d_losses_experiments/3d_lowres_Dice_loss_summary.json',
#     'Focal Loss': 'nnUNet/results/3d_losses_experiments/3d_lowres_nnUNetTrainerFocalLoss_summary.json',
#     'Tversky Loss': 'nnUNet/results/3d_losses_experiments/3d_lowres_Tversky_loss_summary.json'
# }

# json_files = {
#     'Elastic augmentation': 'nnUNet/results/3d_augmentations_experiments/3d_lowres_Dataset058_SegTHOR_Elastic_Dice_CE_loss_summary.json',
#     'Noise augmentation': 'nnUNet/results/3d_augmentations_experiments/3d_lowres_Dataset059_SegTHOR_Noise_Dice_CE_loss_summary.json',
#     'Affine augmentation': 'nnUNet/results/3d_augmentations_experiments/3d_lowres_Dataset057_SegTHOR_Affine_Dice_CE_loss_summary.json',
#     'All augmentations': 'nnUNet/results/3d_augmentations_experiments/3d_lowres_Dataset060_SegTHOR_Combined_Dice_CE_loss_summary.json',
#     'Baseline 3d nnUNet': 'nnUNet/results/3d_augmentations_experiments/3d_lowres_Dataset055_SegTHOR_Corrected_GT_Dice_CE_loss_summary.json',
#     'Baseline 2d nnUNet' : 'nnUNet/results/loss_experiments/corrected_Haussdorff/Corrected_GT_Dice_CE_loss_summary.json'
# }

json_files = {
    'CRF': 'nnUNet/results/CRF_experiments/CRF_output/3d/CRF_summary.json',
    'Baseline 3d nnUNet': 'nnUNet/results/3d_augmentations_experiments/3d_lowres_Dataset055_SegTHOR_Corrected_GT_Dice_CE_loss_summary.json',
    'Baseline 2d nnUNet': 'nnUNet/results/loss_experiments/corrected_Haussdorff/Corrected_GT_Dice_CE_loss_summary.json'
}

# Prepare data for plotting
all_data = []

for model_name, json_file in json_files.items():
    # Load the json file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # For each metric
    for metric_name in ['Dice', 'Hausdorff', 'Hausdorff95', 'IoU']:
        # Collect metric values per class across all cases
        metric_values_per_class = {}  # Key: class_idx, Value: list of values
        for case in data['metric_per_case']:
            metrics = case['metrics']
            for class_idx_str, class_metrics in metrics.items():
                if class_idx_str == '0':
                    continue  # Skip background
                class_idx = int(class_idx_str)
                class_name = class_names[class_idx]
                value = class_metrics.get(metric_name, None)
                if value is not None:
                    if class_idx not in metric_values_per_class:
                        metric_values_per_class[class_idx] = []
                    metric_values_per_class[class_idx].append(value)
        # Now, add the data to all_data
        for class_idx, values in metric_values_per_class.items():
            class_name = class_names[class_idx]
            for value in values:
                all_data.append({
                    'Model': model_name,
                    'Class': class_name,
                    'Metric': metric_name,
                    'Value': value
                })

# Create a DataFrame
df = pd.DataFrame(all_data)

os.makedirs(output_dir, exist_ok=True)

# Create boxplots for each metric using custom themes
for theme in ['light', 'dark']:
    if theme == 'dark':
        set_custom_dark_theme()
    else:
        set_custom_light_theme()
    for metric_name in df['Metric'].unique():
        plt.figure(figsize=(12, 8))
        sns.boxplot(
            data=df[df['Metric'] == metric_name],
            x='Class',
            y='Value',
            hue='Model',
            palette=model_palette,
            showfliers=False  # Hide outliers for clarity
        )
        plt.title(f"nnUNet {network_dim}d {metric_name} per class for different models", fontsize=14)
        plt.ylabel(metric_name, fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.xticks(rotation=45, fontsize=12)
        plt.legend(title='Model', fontsize=12, title_fontsize=12)
        plt.ylim(0, None)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric_name}_per_class_{theme}.png"), dpi=300)
        plt.show()