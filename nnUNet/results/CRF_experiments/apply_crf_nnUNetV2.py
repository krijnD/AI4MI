import argparse
import numpy as np
import SimpleITK as sitk
import os
import pydensecrf.densecrf as dcrf
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Apply CRF to nnUNet predictions.')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing the original images.')
    parser.add_argument('--prediction_dir', type=str, required=True,
                        help='Directory with nnUNet predictions.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save CRF-processed outputs.')
    parser.add_argument('--gt_dir', type=str,
                        help='Directory containing the ground truth segmentations (optional).')
    parser.add_argument('--patient_ids', nargs='+',
                        help='List of patient IDs to process. If not specified, all patients in prediction_dir will be processed.')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of segmentation classes.')
    # CRF parameters
    parser.add_argument('--t', type=int, default=5,
                        help='Number of iterations for CRF inference.')
    parser.add_argument('--sxy_gaussian', type=float, default=3,
                        help='CRF Gaussian pairwise potential sxy parameter.')
    parser.add_argument('--compat_gaussian', type=float, default=3,
                        help='CRF Gaussian pairwise potential compatibility parameter.')
    parser.add_argument('--sxy_bilateral', type=float, default=80,
                        help='CRF Bilateral pairwise potential sxy parameter.')
    parser.add_argument('--srgb_bilateral', type=float, default=13,
                        help='CRF Bilateral pairwise potential srgb parameter.')
    parser.add_argument('--compat_bilateral', type=float, default=10,
                        help='CRF Bilateral pairwise potential compatibility parameter.')
    args = parser.parse_args()
    return args

def load_patient_data(patient_id, image_dir, prediction_dir):
    # Adjusted to match the image file naming convention
    image_filename = f'{patient_id}_0000.nii.gz'  # Images have '_0000' suffix
    image_path = os.path.join(image_dir, image_filename)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image file not found: {image_path}')
    image = sitk.ReadImage(image_path)
    image_np = sitk.GetArrayFromImage(image)  # Shape: (D, H, W)

    # Load the predicted probabilities
    npz_filename = f'{patient_id}.npz'  # Predictions are saved without '_0000'
    npz_path = os.path.join(prediction_dir, npz_filename)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f'Prediction file not found: {npz_path}')
    probs_npz = np.load(npz_path)

    # Use the 'probabilities' key
    if 'probabilities' in probs_npz:
        probs = probs_npz['probabilities']  # Shape: (C, D, H, W)
        print(f"Using key 'probabilities' for patient {patient_id}.")
    else:
        raise KeyError(f"'probabilities' key not found in {npz_filename}. Available keys: {probs_npz.files}")

    return image, image_np, probs

def crf_inference(image_slice, probs_slice, num_classes, 
                  t=5, sxy_gaussian=3, compat_gaussian=3, 
                  sxy_bilateral=80, srgb_bilateral=13, compat_bilateral=10):
    h, w = image_slice.shape[:2]
    d = dcrf.DenseCRF2D(w, h, num_classes)

    # Normalize and reshape probabilities
    U = -np.log(probs_slice + 1e-10)  # Avoid log(0)
    U = U.reshape((num_classes, -1))
    d.setUnaryEnergy(U.astype(np.float32))

    # Add pairwise potentials
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian)
    d.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb_bilateral, 
                           rgbim=image_slice, compat=compat_bilateral)

    # Inference
    Q = d.inference(t)
    Q = np.array(Q).reshape((num_classes, h, w))
    return Q

def apply_crf_to_volume(image_np, probs, num_classes=5, t=5, sxy_gaussian=3, compat_gaussian=3, sxy_bilateral=80, srgb_bilateral=13, compat_bilateral=10):
    D = image_np.shape[0]
    crf_output = np.zeros_like(probs)

    for i in range(D):
        image_slice = image_np[i, :, :]  # (H, W)
        probs_slice = probs[:, i, :, :]  # (C, H, W)

        # Normalize image slice to [0, 255]
        image_slice_norm = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
        image_slice_uint8 = (image_slice_norm * 255).astype(np.uint8)
        image_slice_rgb = cv2.cvtColor(image_slice_uint8, cv2.COLOR_GRAY2RGB)

        # Apply CRF
        crf_probs_slice = crf_inference(image_slice_rgb, probs_slice, num_classes, 
                                        t=t, sxy_gaussian=sxy_gaussian, compat_gaussian=compat_gaussian, 
                                        sxy_bilateral=sxy_bilateral, srgb_bilateral=srgb_bilateral, compat_bilateral=compat_bilateral)
        crf_output[:, i, :, :] = crf_probs_slice

    return crf_output

def save_crf_output(crf_probs, original_image, output_path):
    # Get the segmentation labels
    crf_segmentation = np.argmax(crf_probs, axis=0)  # Shape: (D, H, W)

    # Convert to SimpleITK image
    crf_segmentation_image = sitk.GetImageFromArray(crf_segmentation.astype(np.uint8))
    crf_segmentation_image.CopyInformation(original_image)

    # Save the image
    sitk.WriteImage(crf_segmentation_image, output_path)

def compute_dice_coefficient(ground_truth, prediction, num_classes):
    dices = []
    for class_idx in range(1, num_classes):  # Exclude background
        gt_class = (ground_truth == class_idx)
        pred_class = (prediction == class_idx)
        intersection = np.logical_and(gt_class, pred_class).sum()
        union = gt_class.sum() + pred_class.sum()
        dice = (2. * intersection) / (union + 1e-8)
        dices.append(dice)
    return dices

def process_patient(patient_id, image_dir, prediction_dir, output_dir, gt_dir=None, num_classes=5, t=5, sxy_gaussian=3, compat_gaussian=3, sxy_bilateral=80, srgb_bilateral=13, compat_bilateral=10):
    # Load data
    try:
        image, image_np, probs = load_patient_data(patient_id, image_dir, prediction_dir)
    except FileNotFoundError as e:
        print(e)
        return

    # Apply CRF
    crf_probs = apply_crf_to_volume(image_np, probs, num_classes=num_classes, t=t, sxy_gaussian=sxy_gaussian,
                                    compat_gaussian=compat_gaussian, sxy_bilateral=sxy_bilateral,
                                    srgb_bilateral=srgb_bilateral, compat_bilateral=compat_bilateral)

    # Save output
    output_path = os.path.join(output_dir, f'{patient_id}.nii.gz')
    save_crf_output(crf_probs, image, output_path)
    print(f'Saved CRF-processed segmentation for {patient_id} to {output_path}')

    # Optionally compute Dice coefficient if ground truth is provided
    if gt_dir:
        gt_path = os.path.join(gt_dir, f'{patient_id}.nii.gz')
        if not os.path.exists(gt_path):
            print(f'Ground truth file not found: {gt_path}')
            return
        gt_image = sitk.ReadImage(gt_path)
        gt_np = sitk.GetArrayFromImage(gt_image)
        crf_segmentation = np.argmax(crf_probs, axis=0)
        dices = compute_dice_coefficient(gt_np, crf_segmentation, num_classes)
        print(f'Dice coefficients for {patient_id}:')
        for idx, dice in enumerate(dices):
            print(f'  Class {idx+1}: {dice:.4f}')
    else:
        dices = None

    return dices

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.patient_ids:
        patient_ids = args.patient_ids
    else:
        # Get patient IDs from prediction directory
        # Assuming files are named as 'Patient_01.npz', extract 'Patient_01'
        patient_ids = [os.path.splitext(f)[0] for f in os.listdir(args.prediction_dir) if f.endswith('.npz')]
        patient_ids = list(set(patient_ids))  # Remove duplicates

    all_dices = []

    for patient_id in patient_ids:
        print(f'Processing {patient_id}')
        dices = process_patient(
            patient_id,
            args.image_dir,
            args.prediction_dir,
            args.output_dir,
            gt_dir=args.gt_dir,
            num_classes=args.num_classes,
            t=args.t,
            sxy_gaussian=args.sxy_gaussian,
            compat_gaussian=args.compat_gaussian,
            sxy_bilateral=args.sxy_bilateral,
            srgb_bilateral=args.srgb_bilateral,
            compat_bilateral=args.compat_bilateral
        )
        if dices:
            all_dices.append((patient_id, dices))

    # Optionally, print or save overall metrics
    if all_dices:
        num_classes = args.num_classes - 1  # Excluding background
        sum_dices = np.zeros(num_classes)
        for patient_id, dices in all_dices:
            sum_dices += np.array(dices)
        mean_dices = sum_dices / len(all_dices)
        print('Mean Dice coefficients over all patients:')
        for idx, dice in enumerate(mean_dices):
            print(f'  Class {idx+1}: {dice:.4f}')

if __name__ == '__main__':
    main()

# import numpy as np
# import SimpleITK as sitk
# import os

# npz_path = '/home/kdignumsepu/AI4MI/data/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Dataset055_SegTHOR_Corrected_GT/predictions_with_probs_Dice_CE_loss/Patient_38.npz'
# probs_npz = np.load(npz_path)
# print(probs_npz.files)

# probabilities = probs_npz['probabilities']

# print(f"Probabilities shape: {probabilities.shape}")
# print(f"Min value: {probabilities.min()}, Max value: {probabilities.max()}")
# print(f"Sum over classes (should be close to 1): {np.mean(np.sum(probabilities, axis=0))}")

# image_path = "/home/kdignumsepu/AI4MI/data/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Dataset055_SegTHOR_Corrected_GT/imagesTs/Patient_38_0000.nii.gz"
# if not os.path.exists(image_path):
#     raise FileNotFoundError(f'Image file not found: {image_path}')
# image = sitk.ReadImage(image_path)
# image_np = sitk.GetArrayFromImage(image)  # Shape: (D, H, W)

# print(f"Image shape: {image_np.shape}")
# print(f"Probabilities shape: {probabilities.shape}")