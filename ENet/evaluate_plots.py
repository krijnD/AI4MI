import pandas as pd
import torch
import pyvista as pv
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt

from matplotlib.colors import ListedColormap

custom_palette = {
    'roze': '#eb8fd8',
    'groen': '#b9d4b4',
    'paars': '#ba94e9',
    'blue': '#4C8BE2',
    'orange': '#E27A3F',
    'grey_light': '#1F3240',
    'grey_dark': '#16242F'
}

custom_cmap = ListedColormap([
    custom_palette['grey_dark'],  # Class 1
    custom_palette['groen'],  # Class 2
    custom_palette['paars'],  # Class 3
    custom_palette['blue'],  # Class 4
    custom_palette['orange']  # Class 5
])

class_names = ['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta']


def set_custom_dark_theme():
    # Set light grey background
    sns.set_context('notebook', font_scale=1.2)
    sns.set_style({
        'axes.facecolor': custom_palette['grey_light'],  # Set background to light grey
        'axes.edgecolor': 'white',  # Edge color of the plot
        'axes.labelcolor': 'white',  # Axis labels color
        'xtick.color': 'white',  # X-tick color
        'ytick.color': 'white',  # Y-tick color
        'grid.color': 'white',  # Gridline color
        'figure.facecolor': custom_palette['grey_light'],  # Set figure background to light grey
        'text.color': 'white'  # Color of text in the plot
    })

    # Set the color palette for seaborn plots (for lines)
    sns.set_palette([custom_palette['roze'],
                     custom_palette['groen'],
                     custom_palette['paars'],
                     custom_palette['blue'],
                     custom_palette['orange']])


def set_custom_light_theme():
    # Set light grey background
    sns.set_context('notebook', font_scale=1.2)
    sns.set_style({
        'axes.facecolor': 'white',  # Set background to light grey
        'axes.edgecolor': 'black',  # Edge color of the plot
        'axes.labelcolor': 'black',  # Axis labels color
        'xtick.color': 'black',  # X-tick color
        'ytick.color': 'black',  # Y-tick color
        'grid.color': custom_palette["grey_light"],  # Gridline color
        'figure.facecolor': 'white',  # Set figure background to light grey
        'text.color': 'black'  # Color of text in the plot
    })

    # Set the color palette for seaborn plots (for lines)
    sns.set_palette([custom_palette['roze'],
                     custom_palette['groen'],
                     custom_palette['paars'],
                     custom_palette['blue'],
                     custom_palette['orange']])


def plot_results(image, plottables, idx, evaluate_dir):
    for theme in ["light", "dark"]:
        if theme == "dark":
            set_custom_dark_theme()
        else:
            set_custom_light_theme()

        img_np = image.cpu().numpy()[0, 0, :, :]  # Shape: (H, W)
        fig, axs = plt.subplots(1, len(plottables) + 1, figsize=(20, 5))

        axs_i = 0
        axs[axs_i].imshow(img_np, cmap='gray')
        axs[axs_i].set_title('Input Image')
        axs[axs_i].axis('off')
        axs_i += 1

        for name, prediction in plottables.items():
            pred_class = torch.argmax(prediction, dim=1)  # Shape: (B, H, W)
            prediction_np = pred_class.cpu().numpy()[0]  # Shape: (H, W)

            axs[axs_i].imshow(prediction_np, cmap=custom_cmap, vmin=0, vmax=5 - 1)
            axs[axs_i].set_title(name)
            axs[axs_i].axis('off')

            axs_i += 1

        plt.savefig(evaluate_dir + "/predict_" + str(idx) + "_" + theme + ".png")
        plt.close()


def plot_metrics(metrics, evaluate_dir):
    for theme in ["light", "dark"]:
        if theme == "dark":
            set_custom_dark_theme()
        else:
            set_custom_light_theme()

        colors = [custom_palette['roze'],
                  custom_palette['groen'],
                  custom_palette['paars'],
                  custom_palette['blue']]

        for name_score, values in metrics.items():
            scores = np.array(values)  # Shape: (num_samples, num_classes)

            # Skip the background class using slicing
            data_to_plot = [scores[:, i] for i in range(1, len(class_names))]

            # Prepare data in long-form DataFrame for seaborn
            data = []
            for i, class_name in enumerate(class_names[1:]):
                for value in data_to_plot[i]:
                    data.append({'Class': class_name, name_score: value})

            df = pd.DataFrame(data)

            plt.figure(figsize=(8, 6))

            # Boxplot without outliers
            sns.boxplot(x='Class', y=name_score, data=df, palette=colors, showfliers=False)  # No outliers
            # Updated plotting function
            sns.boxplot(x='Class', y=name_score, data=df, hue='Class', palette=colors, showfliers=False, dodge=False)
            plt.legend([], [], frameon=False)  # Suppress the legend

            # Optional stripplot (can be removed if you want no dots)
            # sns.stripplot(x='Class', y=name_score, data=df, color='black', alpha=0.5, jitter=0.2)

            plt.title(f"{name_score} per class (excluding Background)", fontsize=14)
            plt.ylabel(name_score, fontsize=12)
            plt.xlabel('Class', fontsize=12)

            plt.xticks(rotation=45, fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(evaluate_dir, f"{name_score}_per_class_{theme}.png"), dpi=300)
            plt.close()


def animate_3d_volume(volume_predictions, volume_ground_truths, evaluate_dir, args):
    # Assuming volume_predictions and volume_ground_truths are defined as before
    ids = [1, 13, 22, 28, 30]
    for volume_id in ids:
        # Retrieve the list of slices for the given volume_id
        pred_slices = volume_predictions[volume_id]  # List of (slice_idx, pred_slice)
        gt_slices = volume_ground_truths[volume_id]  # List of (slice_idx, gt_slice)

        # Sort the slices by slice_idx
        pred_slices_sorted = sorted(pred_slices, key=lambda x: x[0])
        gt_slices_sorted = sorted(gt_slices, key=lambda x: x[0])

        # Stack the slices into a Tensor volume
        pred_volume = torch.stack([slice_data for idx, slice_data in pred_slices_sorted], dim=1)
        gt_volume = torch.stack([slice_data for idx, slice_data in gt_slices_sorted], dim=1)

        # Now pred_volume and gt_volume are Tensors of shape (K, D, H, W)
        # where K is the number of classes, D is the depth (number of slices)

        # Call the rendering function
        # render_3d_segmentation(pred_volume, gt_volume, volume_id)
        render_3d_segmentation(pred_volume, gt_volume, volume_id, args, output_dir=evaluate_dir)


def render_3d_segmentation(pred_volume, gt_volume, volume_id, args, output_dir):
    # pred_volume: Tensor of shape (K, D, H, W)
    # gt_volume: Tensor of shape (K, D, H, W)

    # Convert to numpy arrays and get label volumes
    pred_volume_np = torch.argmax(pred_volume, dim=0).cpu().numpy()  # Shape: (D, H, W)
    gt_volume_np = torch.argmax(gt_volume, dim=0).cpu().numpy()

    # Transpose the volumes to match PyVista's (X, Y, Z) format
    pred_volume_np = np.transpose(pred_volume_np, (2, 1, 0))  # Now shape is (W, H, D)
    gt_volume_np = np.transpose(gt_volume_np, (2, 1, 0))

    # Create the grids
    grid_pred = pv.UniformGrid()
    grid_pred.dimensions = pred_volume_np.shape
    grid_pred.spacing = (1, 1, 1)
    grid_pred.origin = (0, 0, 0)

    grid_gt = pv.UniformGrid()
    grid_gt.dimensions = gt_volume_np.shape
    grid_gt.spacing = (1, 1, 1)
    grid_gt.origin = (0, 0, 0)

    # Assign the label data to 'point_data'
    grid_pred.point_data["labels"] = pred_volume_np.flatten(order="F")
    grid_gt.point_data["labels"] = gt_volume_np.flatten(order="F")

    # Define class names and colors (adjust as needed)
    class_names = ['Background', 'Esophagus', 'Heart', 'Trachea', 'Aorta']
    class_colors = [
        'gray',  # Background
        custom_palette['roze'],  # Esophagus
        custom_palette['groen'],  # Heart
        custom_palette['paars'],  # Trachea
        custom_palette['blue']  # Aorta
    ]

    # Create the plotter
    p = pv.Plotter(shape=(1, 2), window_size=(1600, 800), off_screen=True)

    # Function to add class surfaces
    def add_class_surfaces(grid, subplot_index, title):
        p.subplot(0, subplot_index)
        for c in range(1, len(class_names)):  # Skip background if desired
            class_label = c
            # Threshold the grid to extract the class
            class_grid = grid.threshold(value=(class_label - 0.1, class_label + 0.1), scalars='labels')
            if class_grid.n_points == 0:
                continue  # Skip if no points for this class
            # Extract the surface mesh
            surface = class_grid.contour(isosurfaces=[class_label], scalars='labels')
            # Add the mesh to the plotter
            p.add_mesh(surface, color=class_colors[c], opacity=0.6, label=class_names[c])
        p.add_legend(bcolor='white')
        p.add_axes()
        p.set_background('white')
        p.add_title(title)

    # Add ground truth surfaces
    add_class_surfaces(grid_gt, subplot_index=0, title='Ground Truth' if not args.crf else 'Normal Prediction')

    # Add prediction surfaces
    add_class_surfaces(grid_pred, subplot_index=1, title='Prediction' if not args.crf else 'CRF Prediction')

    # Link the views
    p.link_views()

    # Define the camera positions you want to capture
    camera_positions = {
        'isometric': 'iso',
        'front': 'xz',
        'side': 'yz',
        'top': 'xy',
    }

    # Save a screenshot for each camera position
    for view_name, camera_pos in camera_positions.items():
        # Set the camera position
        p.camera_position = camera_pos
        # Update rendering
        p.render()
        # Save the rendering to a file
        screenshot_path = os.path.join(output_dir, f'Patient_{volume_id}_{view_name}_view.png')
        p.screenshot(screenshot_path)
        print(f"Saved {view_name} view to {screenshot_path}")

    # Close the plotter
    p.close()
    print(f"Rendered 3D segmentation saved to {screenshot_path}")
