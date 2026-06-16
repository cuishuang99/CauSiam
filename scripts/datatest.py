"""Dataset sanity checks (run from project root: python scripts/datatest.py)."""

import os.path as osp
from glob import glob

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.mydataset import Mydataset
from utils.img_utils import path2tensorimg, save_tensor2img


def check_data_size(directory):
    imgs = sorted(glob(osp.join(directory, '*.*')))
    img = path2tensorimg(imgs[0])
    height, width = img.shape[1], img.shape[2]
    for image_path in tqdm(imgs):
        img = path2tensorimg(image_path)
        if img.shape[1] != height or img.shape[2] != width:
            return height, width, False
    return height, width, True


def check_data_channel(directory):
    imgs = sorted(glob(osp.join(directory, '*.*')))
    img = path2tensorimg(imgs[0])
    channels = img.shape[0]
    for image_path in tqdm(imgs):
        img = path2tensorimg(image_path)
        if img.shape[0] != channels:
            return channels, False
    return channels, True


def img_minsize(directory):
    imgs = sorted(glob(osp.join(directory, '*.*')))
    min_size = torch.inf
    for image_path in tqdm(imgs):
        img = path2tensorimg(image_path)
        min_size = min(min_size, img.shape[1], img.shape[2])
    return min_size


def check_data_pair(input_dir, gt_dir, crop_h, crop_w):
    dataset = Mydataset('train', input_dir, gt_dir, crop_h, crop_w)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    data = next(iter(dataloader))
    save_tensor2img(data['input'][0], 'input.png')
    save_tensor2img(data['gt'][0], 'gt.png')


def check_data(input_dir, gt_dir):
    height, width, same_size = check_data_size(input_dir)
    channels, same_channel = check_data_channel(input_dir)
    print('Data Same Size: {}'.format(same_size))
    print('Data Same Channel: {}'.format(same_channel))
    if same_size:
        print('val_batch_size >= 1')
        print('Data Size HxW: {}x{}'.format(height, width))
        check_data_pair(input_dir, gt_dir, height, width)
    else:
        min_size = img_minsize(input_dir)
        print('val_batch_size = 1')
        print('Max Crop Size: {}'.format(min_size))
        check_data_pair(input_dir, gt_dir, min_size, min_size)


if __name__ == '__main__':
    check_data('/home/csh/dataset/DPDD/train/inputC', '/home/csh/dataset/DPDD/train/target')
