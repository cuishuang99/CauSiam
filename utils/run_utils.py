"""Shared utilities for training, validation, and testing scripts."""

import importlib
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn

LOG_FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s'
LOG_DATE_FORMAT = '%y-%m-%d %H:%M:%S'


def init_logger(name, log_path, mode='a'):
    """Create a logger that writes to both a file and stdout."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def zero_timedelta():
    """Return a zero-length timedelta for elapsed-time tracking."""
    now = datetime.now()
    return now - now


def set_random_seed(seed, gpu):
    """Configure GPU device and random seeds."""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    # cudnn.deterministic = True  # enable for full reproducibility


def parse_option_args(option='dpdd', gpu='0'):
    """Parse CLI arguments using the dataset-specific option module."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, default=option)
    parser.add_argument('--gpu', type=str, default=gpu)
    opt, _ = parser.parse_known_args()
    option_module = importlib.import_module('options.' + opt.option + '.option')
    parser = option_module.get_option(parser)
    return parser.parse_args()


def format_args(args):
    """Format an argparse Namespace as a log-friendly string."""
    lines = ['Parser:']
    for key, value in vars(args).items():
        lines.append('{:<20}: {}'.format(key, value))
    return '\n'.join(lines) + '\n'


def log_psnr(psnr, best_psnr, logger):
    """Log validation PSNR and return the updated best score."""
    msg = 'Validation PSNR : {:8.4f}'.format(psnr)
    if psnr > best_psnr:
        best_psnr = psnr
        msg += ' best psnr : {:8.4f}'.format(best_psnr)
    logger.info(msg)
    return best_psnr
