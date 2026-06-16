"""Training script with optional inline validation."""

import argparse
import importlib
import os
import os.path as osp
import shutil
import time
from datetime import datetime
from glob import glob

import torch as T
from torch.utils.data import DataLoader

from model.model import Model
from utils.eval_utils import build_dataloader, create_dataset, run_evaluation
from utils.img_utils import create_dir, save_com_img
from utils.run_utils import format_args, init_logger, set_random_seed, zero_timedelta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, default='dpdd')
    parser.add_argument('--gpu', type=str, default='2,3,5')
    opt, _ = parser.parse_known_args()
    option_module = importlib.import_module('options.' + opt.option + '.option')
    parser = option_module.get_option(parser)
    args = parser.parse_args()
    args.phase = 'train'
    return args


def log_train_init(train_logger, args, train_dataset, model, start_epoch):
    train_logger.info(format_args(args))
    train_logger.info('PID: {}'.format(os.getpid()))
    train_logger.info('Wandb id: {}'.format(args.wandb_id))
    train_logger.info('gpu id: {}'.format(args.gpu))
    train_logger.info('Number of train images: {}'.format(len(train_dataset)))
    train_logger.info('Start from epoch: {}'.format(start_epoch))
    train_logger.info('Total epochs: {}'.format(args.epochs))
    train_logger.info('Parameters: {:.3f} M'.format(model.cal_parameters() / 1e6))
    train_logger.info('FLOPs: {:.4f} G'.format(model.cal_floaps() / 1e9))
    train_logger.info('START'.center(51, '*'))


def log_val_init(val_logger, args, val_dataset, start_epoch, best_psnr):
    val_logger.info(format_args(args))
    val_logger.info('PID: {}'.format(os.getpid()))
    val_logger.info('gpu id: {}'.format(args.gpu))
    val_logger.info('Number of val images: {}'.format(len(val_dataset)))
    val_logger.info('Start from epoch: {}'.format(start_epoch))
    val_logger.info('Total epochs: {}'.format(args.epochs))
    if start_epoch != 0:
        val_logger.info('best psnr: {:.4f}'.format(best_psnr))
    val_logger.info('START'.center(51, '*'))


def log_loss(train_logger, epoch, loss_dict):
    msg = 'epoch: {:4d}'.format(epoch)
    for key, value in loss_dict.items():
        msg += ' {:s}: {:8.4f}'.format(key, value)
    train_logger.info(msg)


def snapshot_training_code(train_snapshot_root, args):
    pro_dir = os.path.dirname(__file__)
    code_root = osp.join(train_snapshot_root, 'code')
    shutil.copytree(osp.join(pro_dir, 'loss'), osp.join(code_root, 'loss'))
    shutil.copytree(osp.join(pro_dir, 'model'), osp.join(code_root, 'model'))
    shutil.copytree(
        osp.join(pro_dir, 'options', args.option),
        osp.join(code_root, 'options'),
    )
    shutil.copy(
        osp.join(pro_dir, 'options', 'model_para.py'),
        osp.join(code_root, 'options'),
    )


def main():
    args = parse_args()
    assert args.val_freq % args.train_freq == 0

    set_random_seed(args.seed, args.gpu)
    model = Model(args)

    train_dataset = create_dataset(
        args, 'train', args.train_input_dir, args.train_gt_dir, args.cropH, args.cropW
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    if args.resume_train_dir is not None:
        train_snapshot_root = args.resume_train_dir
        model_dir = osp.join(train_snapshot_root, 'model_state')
        train_dir = osp.join(train_snapshot_root, 'train_state')
        model_list = sorted(
            glob(osp.join(model_dir, '*.*')),
            key=lambda x: int(osp.splitext(osp.basename(x))[0].split('_')[1])
            if osp.splitext(osp.basename(x))[0] != 'best' else -1,
            reverse=True,
        )
        train_list = sorted(
            glob(osp.join(train_dir, '*.*')),
            key=lambda x: int(osp.splitext(osp.basename(x))[0].split('_')[1]),
            reverse=True,
        )
        final_train = train_list[0]
        model.load_model_state(model_list[0])
        model.load_train_state(final_train)
        train_state = T.load(final_train)
        start_epoch = train_state['epoch'] + 1
        train_du = train_state['train_du']
    else:
        train_snapshot_root = osp.join(
            args.save, 'train-{}'.format(time.strftime('%Y%m%d-%H%M%S'))
        )
        model_dir = osp.join(train_snapshot_root, 'model_state')
        train_dir = osp.join(train_snapshot_root, 'train_state')
        create_dir(model_dir)
        create_dir(train_dir)
        start_epoch = 0
        train_du = zero_timedelta()
        snapshot_training_code(train_snapshot_root, args)

    train_logger = init_logger('train', osp.join(train_snapshot_root, 'train.log'))
    log_train_init(train_logger, args, train_dataset, model, start_epoch)

    val_logger = None
    if args.val_flag:
        val_snapshot_root = osp.join(
            train_snapshot_root,
            osp.basename(train_snapshot_root).replace('train', 'val'),
        )
        if args.resume_train_dir is not None:
            assert osp.exists(val_snapshot_root)
            img_root = osp.join(val_snapshot_root, 'val_imgs')
            index_dir = osp.join(val_snapshot_root, 'index_state')
            index_list = sorted(
                glob(osp.join(index_dir, '*.*')),
                key=lambda x: int(osp.splitext(osp.basename(x))[0].rsplit('_')[1]),
                reverse=True,
            )
            index_state = model.load_index_state(index_list[0])
            best_psnr = index_state['best_psnr']
            val_du = index_state['val_du']
        else:
            index_dir = osp.join(val_snapshot_root, 'index_state')
            img_root = osp.join(val_snapshot_root, 'val_imgs')
            create_dir(index_dir)
            create_dir(img_root)
            best_psnr = 0
            val_du = zero_timedelta()

        val_dataset = create_dataset(args, 'val', args.val_input_dir, args.val_gt_dir)
        val_dataloader = build_dataloader(val_dataset, args)
        val_logger = init_logger('val', osp.join(val_snapshot_root, 'val.log'))
        log_val_init(val_logger, args, val_dataset, start_epoch, best_psnr)

    for epoch in range(start_epoch, args.epochs):
        if epoch % args.train_freq == 0:
            train_st = datetime.now()

        for train_data in train_dataloader:
            model.feed_data(train_data)
            loss_dict = model.optimize_parameters()

        log_loss(train_logger, epoch, loss_dict)
        model.scheduler.step()

        if (epoch + 1) % args.train_freq == 0:
            train_t = datetime.now() - train_st
            train_du += train_t

        if (epoch + 1) % args.val_freq == 0 and args.val_flag:
            val_logger.info('epoch: {:4d}'.format(epoch))
            img_dir = osp.join(img_root, 'epoch_{}'.format(epoch))
            create_dir(img_dir)

            prev_best = best_psnr
            best_psnr, val_elapsed, output_img_list, gt_img_list, save_path_list = run_evaluation(
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
                for output, gt, path in zip(output_img_list, gt_img_list, save_path_list):
                    save_com_img(path, output, gt)

            val_du += val_elapsed
            val_logger.info('Time: {} Total time: {}\n'.format(val_elapsed, val_du))
            model.save_index(epoch, index_dir, best_psnr, val_du)
            T.cuda.empty_cache()

        if (epoch + 1) % args.train_freq == 0:
            model.save_model(epoch, model_dir, train_dir, train_du)
            train_logger.info('Saving models and training states.')
            train_logger.info('Time: {} Total time: {}\n'.format(train_t, train_du))

    if args.epochs % args.train_freq != 0:
        train_du += datetime.now() - train_st

    train_logger.info('Training time: {}'.format(train_du))
    train_logger.info('END'.center(51, '*') + '\n')

    if args.val_flag:
        val_logger.info('best psnr: {}'.format(best_psnr))
        val_logger.info('Validation time: {}'.format(val_du))
        val_logger.info('END'.center(51, '*') + '\n')


if __name__ == '__main__':
    main()
