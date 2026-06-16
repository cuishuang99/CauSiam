"""Single-dataset evaluation script."""

import os.path as osp

from utils.eval_utils import (
    build_dataloader,
    build_model,
    create_dataset,
    log_experiment_info,
    run_evaluation,
)
from utils.img_utils import create_dir
from utils.run_utils import init_logger, parse_option_args, set_random_seed, zero_timedelta

SNAPSHOT_ROOT = 'result'
IMG_SUBDIR = 'DOF'


def main():
    args = parse_option_args(option='dpdd', gpu='0')
    args.phase = 'val'

    img_root = osp.join(SNAPSHOT_ROOT, IMG_SUBDIR)
    test_log_path = osp.join(SNAPSHOT_ROOT, 'test.log')
    create_dir(img_root)

    logger = init_logger(args.phase, test_log_path)
    set_random_seed(args.seed, args.gpu)

    model = build_model(args)
    val_dataset = create_dataset(args, 'val', args.val_input_dir, args.val_gt_dir)
    val_dataloader = build_dataloader(val_dataset, args)

    log_experiment_info(logger, args, val_dataset, model)

    total_elapsed = zero_timedelta()
    _, val_elapsed, _, _, _ = run_evaluation(
        model, val_dataloader, img_root, args, logger
    )
    total_elapsed += val_elapsed
    logger.info('Time: {} Total time: {}\n'.format(val_elapsed, total_elapsed))
    logger.info('End of testing.')


if __name__ == '__main__':
    main()
