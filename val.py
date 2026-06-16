"""Standalone validation script for checkpoints produced during training."""

import argparse
import importlib
import os.path as osp
import time
from glob import glob

import torch as T

from model.model import Model
from utils.eval_utils import (
    build_dataloader,
    create_dataset,
    log_validation_header,
    run_evaluation,
)
from utils.img_utils import create_dir, save_tensor2img
from utils.run_utils import init_logger, set_random_seed, zero_timedelta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, default='lolv1')
    parser.add_argument('--gpu', type=str, required=True)
    opt, _ = parser.parse_known_args()
    option_module = importlib.import_module('options.' + opt.option + '.option')
    parser = option_module.get_option(parser)
    args = parser.parse_args()
    args.phase = 'val'
    return args


def main():
    args = parse_args()
    set_random_seed(args.seed, args.gpu)
    model = Model(args)

    train_snapshot_root = args.train_snapshot_root
    model_dir = osp.join(train_snapshot_root, 'model_state')
    train_dir = osp.join(train_snapshot_root, 'train_state')

    val_dataset = create_dataset(args, 'val', args.val_input_dir, args.val_gt_dir)
    val_dataloader = build_dataloader(val_dataset, args)

    if args.resume_val_dir is not None:
        val_snapshot_root = args.resume_val_dir
        img_root = osp.join(val_snapshot_root, 'val_imgs')
        index_dir = osp.join(val_snapshot_root, 'index_state')
        index_list = sorted(
            glob(osp.join(index_dir, '*.*')),
            key=lambda x: int(osp.splitext(osp.basename(x))[0].rsplit('_')[1]),
            reverse=True,
        )
        index_state = model.load_index_state(index_list[0])
        best_psnr = index_state['best_psnr']
        start_epoch = index_state['epoch'] + 1
        val_du = index_state['val_du']
    else:
        val_snapshot_root = osp.join(
            train_snapshot_root,
            'val-{}-{}'.format(time.strftime('%Y%m%d-%H%M%S'), args.data_name),
        )
        index_dir = osp.join(val_snapshot_root, 'index_state')
        img_root = osp.join(val_snapshot_root, 'val_imgs')
        create_dir(index_dir)
        create_dir(img_root)
        best_psnr = 0
        start_epoch = 0
        val_du = zero_timedelta()

    val_logger = init_logger('val', osp.join(val_snapshot_root, 'val.log'))
    log_validation_header(val_logger, args, val_dataset, start_epoch, best_psnr)

    for epoch in range(start_epoch, args.epochs):
        if (epoch + 1) % args.val_freq != 0:
            continue

        path_check = osp.join(train_dir, 'epoch_{}.pt'.format(epoch))
        model_path = osp.join(model_dir, 'epoch_{}.pt'.format(epoch))
        while not osp.exists(path_check):
            pass

        model.load_model_state(model_path)
        val_logger.info('epoch: {:4d}'.format(epoch))

        img_dir = osp.join(img_root, 'epoch_{}'.format(epoch))
        create_dir(img_dir)

        prev_best = best_psnr
        best_psnr, val_elapsed, output_img_list, _, save_path_list = run_evaluation(
            model,
            val_dataloader,
            img_dir,
            args,
            val_logger,
            best_psnr=best_psnr,
            save_outputs_flag=False,
            show_progress=False,
            rgb_order=False,
        )
        if best_psnr > prev_best:
            model.save_best_model(model_dir)

        if args.img_save:
            for output, path in zip(output_img_list, save_path_list):
                save_tensor2img(output, path)

        val_du += val_elapsed
        val_logger.info('Time: {} Total time: {}\n'.format(val_elapsed, val_du))
        model.save_index(epoch, index_dir, best_psnr, val_du)
        T.cuda.empty_cache()

    val_logger.info('best psnr: {}'.format(best_psnr))
    val_logger.info('Validation time: {}'.format(val_du))
    val_logger.info('END'.center(51, '*') + '\n')


if __name__ == '__main__':
    main()
