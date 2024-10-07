#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import SimpleITK as sitk
import argparse
import os

def convert_for_submission(source_dir, target_dir):
    """
    I believe they want .nii, not .nii.gz
    :param source_dir:
    :param target_dir:
    :return:
    """
    files = subfiles(source_dir, suffix=".nii.gz", join=False)
    maybe_mkdir_p(target_dir)
    for f in files:
        img = sitk.ReadImage(join(source_dir, f))
        out_file = join(target_dir, f[:-7] + ".nii")
        sitk.WriteImage(img, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SEGTHOR data and copy to nnUNet format.")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the SEGTHOR data directory.")
    parser.add_argument('--nnUNet_raw', type=str, required=True, help="Path to nnUNet raw data directory.")
    parser.add_argument('--nnUNet_preprocessed', type=str, required=False, help="Path to nnUNet preprocessed data directory.")
    parser.add_argument('--nnUNet_results', type=str, required=False, help="Path to nnUNet results directory.")
    args = parser.parse_args()

    # Get base directories from arguments
    base = args.data_dir
    nnUNet_raw = args.nnUNet_raw
    # nnUNet_preprocessed = args.nnUNet_preprocessed
    # nnUNet_results = args.nnUNet_results

    task_id = 55
    task_name = "SegTHOR"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_patient_names = []
    test_patient_names = []
    train_patients = subfolders(join(base, "train"), join=False)
    for p in train_patients:
        curr = join(base, "train", p)
        label_file = join(curr, "GT.nii.gz")
        image_file = join(curr, p + ".nii.gz")
        shutil.copy(image_file, join(imagestr, p + "_0000.nii.gz"))
        shutil.copy(label_file, join(labelstr, p + ".nii.gz"))
        train_patient_names.append(p)

    test_patients = subfiles(join(base, "test"), join=False, suffix=".nii.gz")
    for p in test_patients:
        p = p[:-7]
        curr = join(base, "test")
        image_file = join(curr, p + ".nii.gz")
        shutil.copy(image_file, join(imagests, p + "_0000.nii.gz"))
        test_patient_names.append(p)


    json_dict = OrderedDict()
    json_dict['name'] = "SegTHOR"
    json_dict['description'] = "SegTHOR"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "esophagus",
        "2": "heart",
        "3": "trachea",
        "4": "aorta",
    }
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))