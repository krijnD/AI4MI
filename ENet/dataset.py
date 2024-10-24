#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from typing import Callable, Union

import torch
from PIL import Image
from torch.utils.data import Dataset


def make_dataset(root, subset) -> list[dict]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    img_dir = root / subset / 'img'
    gt_dir = root / subset / 'gt'

    images = sorted(img_dir.glob("*.png"))
    labels = sorted(gt_dir.glob("*.png"))

    data_list = []

    for img_file, label_file in zip(images, labels):
        stem = img_file.stem  # e.g., 'Patient_XX_YYYY'
        parts = stem.split('_')  # Splits into ['Patient', 'XX', 'YYYY']
        if len(parts) == 3:
            volume_id = int(parts[1])  # Convert 'XX' to integer
            slice_idx = int(parts[2])  # Convert 'YYYY' to integer
        else:
            raise ValueError(f"Filename {img_file.name} does not match expected pattern 'Patient_XX_YYYY.png'.")

        data_list.append({
            'img': img_file,
            'gt': label_file,
            'volume_id': volume_id,
            'slice_idx': slice_idx
        })

    return data_list

class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[torch.Tensor, int, str]]:
        data_entry = self.files[index]
        img_path = data_entry['img']
        gt_path = data_entry['gt']
        volume_id = data_entry['volume_id']
        slice_idx = data_entry['slice_idx']

        img: torch.Tensor = self.img_transform(Image.open(img_path))
        gt: torch.Tensor = self.gt_transform(Image.open(gt_path))

        _, W, H = img.shape
        K, _, _ = gt.shape
        assert gt.shape == (K, W, H)

        return {
            "images": img,
            "gts": gt,
            "stems": img_path.stem,
            "volume_id": volume_id,
            "slice_idx": slice_idx
        }
