"""Low-level image I/O helpers for dataset preprocessing."""

import os

import cv2
import numpy as np


def read_image(path, norm_val=None):
    if norm_val == (2**16 - 1):
        frame = cv2.imread(path, -1)
        frame = frame / norm_val
        frame = frame[..., ::-1]
    else:
        frame = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        frame = frame / 255.0
    return np.expand_dims(frame, axis=0)


def crop_image(img, val=8):
    shape = img.shape
    if len(shape) == 4:
        _, height, width, _ = shape
        return img[:, 0:height - height % val, 0:width - width % val, :]
    if len(shape) == 3:
        height, width = shape[:2]
        return img[0:height - height % val, 0:width - width % val, :]
    height, width = shape[:2]
    return img[0:height - height % val, 0:width - width % val]


def make_lf_aif_gt_dataset(img_list, directory):
    aif_gt_files = []
    assert os.path.isdir(directory), '%s is not a valid directory' % directory
    for image_path in img_list:
        aif_file = os.path.split(image_path)[-1].split('_ap')[0]
        aif_file_name = os.path.join(directory, aif_file + '.png')
        if os.path.exists(aif_file_name):
            aif_gt_files.append(aif_file_name)
    return aif_gt_files
