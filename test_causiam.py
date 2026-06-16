"""Multi-dataset CAUSIAM CLIP-TTA evaluation script."""

import os.path as osp

import torch

from dataset.mydataset import Mydataset
from utils.eval_utils import (
    build_dataloader,
    build_model,
    log_experiment_info,
    run_evaluation,
    save_outputs,
)
from utils.img_utils import create_dir
from utils.run_utils import init_logger, parse_option_args, set_random_seed, zero_timedelta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_NAMES = ['DPDD', 'RealDOF', 'LFDOF', 'RTF']
DATASET_ROOT = '/home/csh/dataset/Defocus_Deblur_Dataset_Test/'
SNAPSHOT_ROOT = 'result-CAUSIAM-DPDNet'
DEFAULT_OPTION = 'dpdd'
DEFAULT_GPU = '5'
DEFAULT_SEED = 42


def evaluate_dataset(model, args, dataset_name, snapshot_root, total_elapsed):
    img_root = osp.join(snapshot_root, dataset_name)
    test_log_path = osp.join(snapshot_root, 'test.log')
    create_dir(img_root)

    logger = init_logger(args.phase, test_log_path)
    val_dataset = Mydataset(
        'val',
        osp.join(DATASET_ROOT, dataset_name, 'test/input'),
        osp.join(DATASET_ROOT, dataset_name, 'test/target'),
    )
    val_dataloader = build_dataloader(val_dataset, args)

    torch.cuda.empty_cache()
    log_experiment_info(logger, args, val_dataset, model)

    _, val_elapsed, output_img_list, _, save_path_list = run_evaluation(
        model,
        val_dataloader,
        img_root,
        args,
        logger,
        save_outputs_flag=False,
    )
    save_outputs(output_img_list, save_path_list)

    total_elapsed += val_elapsed
    logger.info('Time: {} Total time: {}\n'.format(val_elapsed, total_elapsed))
    logger.info('End of testing.')
    return total_elapsed


def main():
    args = parse_option_args(option=DEFAULT_OPTION, gpu=DEFAULT_GPU)
    args.phase = 'val'
    args.seed = DEFAULT_SEED

    set_random_seed(args.seed, args.gpu)
    model = build_model(args)

    total_elapsed = zero_timedelta()
    for dataset_name in DATASET_NAMES:
        total_elapsed = evaluate_dataset(
            model, args, dataset_name, SNAPSHOT_ROOT, total_elapsed
        )


if __name__ == '__main__':
    main()
