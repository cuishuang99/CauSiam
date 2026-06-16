"""Shared helpers for model loading, inference, and evaluation."""

import os
import os.path as osp
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import Model
from .img_utils import create_dir, save_tensor2img
from .metric_cal import cal_index_imgs
from .run_utils import format_args, log_psnr


def build_model(args):
    """Build the model and optionally load checkpoint weights for TTA."""
    model = Model(args)
    if args.resume_model is not None:
        model.load_model_state(args.resume_model)
    return model


def create_dataset(args, phase, input_dir, gt_dir, crop_h=None, crop_w=None):
    """Instantiate the dataset class configured by ``args.option``."""
    if args.option == 'sid':
        from dataset.sid_dataset import SID_dataset
        if phase == 'train':
            return SID_dataset('train', input_dir, gt_dir, crop_h, crop_w)
        return SID_dataset('val', input_dir, gt_dir)

    from dataset.mydataset import Mydataset
    if phase == 'train':
        return Mydataset('train', input_dir, gt_dir, crop_h, crop_w)
    return Mydataset('val', input_dir, gt_dir)


def build_dataloader(dataset, args, shuffle=False, drop_last=False):
    """Create a DataLoader with project-default settings."""
    return DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def log_experiment_info(logger, args, dataset, model):
    """Log parsed arguments and basic dataset/model statistics."""
    logger.info(format_args(args))
    logger.info('gpu id: {}'.format(args.gpu))
    logger.info('Number of val images: {}'.format(len(dataset)))
    logger.info('Parameters: {:,d}'.format(model.cal_parameters()))


def log_validation_header(logger, args, dataset, start_epoch, best_psnr):
    """Log validation session metadata used by train/val scripts."""
    logger.info(format_args(args))
    logger.info('PID: {}'.format(os.getpid()))
    logger.info('gpu id: {}'.format(args.gpu))
    logger.info('Number of val images: {}'.format(len(dataset)))
    logger.info('Start from epoch: {}'.format(start_epoch))
    logger.info('Total epochs: {}'.format(args.epochs))
    if start_epoch != 0:
        logger.info('best psnr: {:.4f}'.format(best_psnr))
    logger.info('START'.center(51, '*'))


def run_inference(model, dataloader, img_dir, show_progress=True):
    """Run inference and collect outputs, ground truths, and save paths."""
    output_img_list, gt_img_list, save_path_list = [], [], []
    iterator = tqdm(dataloader) if show_progress else dataloader
    for val_data in iterator:
        model.feed_data(val_data)
        output = model.test()
        batch_size, _, _, _ = output.shape
        output_img_list += [output[i] for i in range(batch_size)]
        gt_img_list += [val_data['gt'][i] for i in range(batch_size)]
        save_path_list += [
            osp.join(img_dir, basename + '.png') for basename in val_data['name']
        ]
    return output_img_list, gt_img_list, save_path_list


def save_outputs(output_img_list, save_path_list, show_progress=True):
    """Persist predicted images to disk."""
    pairs = list(zip(output_img_list, save_path_list))
    iterator = tqdm(pairs) if show_progress else pairs
    for img, path in iterator:
        save_tensor2img(img, path)


def evaluate_predictions(output_img_list, gt_img_list, args, logger, best_psnr=0,
                         rgb_order=True):
    """Compute PSNR and log the result."""
    psnr = cal_index_imgs(
        output_img_list, gt_img_list, args.y_channel, rgb_order
    )
    return log_psnr(psnr, best_psnr, logger)


def run_evaluation(model, dataloader, img_dir, args, logger, best_psnr=0,
                   save_outputs_flag=True, show_progress=True, rgb_order=True):
    """Run inference, compute PSNR, optionally save images, and log timing."""
    val_start = datetime.now()
    output_img_list, gt_img_list, save_path_list = run_inference(
        model, dataloader, img_dir, show_progress=show_progress
    )
    best_psnr = evaluate_predictions(
        output_img_list, gt_img_list, args, logger, best_psnr, rgb_order
    )
    if save_outputs_flag:
        save_outputs(output_img_list, save_path_list, show_progress=show_progress)
    val_elapsed = datetime.now() - val_start
    return best_psnr, val_elapsed, output_img_list, gt_img_list, save_path_list
