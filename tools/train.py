# @Time    : 2025/7/30 
# @Author  : glyy
import sys
import os
import glob
import argparse
import datetime
import tqdm
from easydict import EasyDict

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.modeling.models.lightstereo.trainer import Trainer
from cfgs.data_basic import DATA_PATH_DICT


def _parse_ckpt_arg(value: str):
    key = str(value).strip().lower()
    if key == 'best':
        return 'best'
    return int(value)


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # mode
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default='cfgs\lightstereo\lightstereo_s_kitti.yaml', required=False, help='specify the config for training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    # save path
    parser.add_argument('--save_root_dir', type=str, default='./output', help='save root dir for this experiment')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument(
        '--resume_dir',
        type=str,
        default='',
        help='existing run directory to resume (contains ckpt/). Skips timestamp renaming.',
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default=None,
        help="checkpoint to resume: 'best' or an epoch index. Overrides MODEL.CKPT.",
    )
    # dataloader
    parser.add_argument('--workers', type=int, default=7, help='number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='data loader pin memory')

    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)
    if args.ckpt is not None:
        cfgs.MODEL.CKPT = _parse_ckpt_arg(args.ckpt)

    dataset_names = [x.DATASET for x in cfgs.DATA_CONFIG.DATA_INFOS]
    unique_dataset_names = list(set(dataset_names))
    if len(unique_dataset_names) == 1:
        exp_dataset_dir = unique_dataset_names[0]
    else:
        exp_dataset_dir = 'MultiDataset'
    args.exp_group_path = os.path.join(exp_dataset_dir, cfgs.MODEL.NAME)
    args.tag = os.path.basename(args.cfg_file)[:-5]

    for each in cfgs.DATA_CONFIG.DATA_INFOS:
        dataset_name = each.DATASET
        if dataset_name == 'KittiDataset':
            dataset_name = 'KittiDataset15' if 'kitti15' in each.DATA_SPLIT.EVALUATING else 'KittiDataset12'
        print(dataset_name)
        each.DATA_PATH = DATA_PATH_DICT[dataset_name]

    args.run_mode = 'train'
    return args, cfgs


def main():
    args, cfgs = parse_config()
    if args.dist_mode:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        group_rank = int(os.environ["GROUP_RANK"])
    else:
        local_rank = 0
        global_rank = 0
        group_rank = 0

    # env
    torch.cuda.set_device(local_rank)
    if args.fix_random_seed:
        seed = 0 if not args.dist_mode else dist.get_rank()
        common_utils.set_random_seed(seed=seed)

    resume_requested = common_utils.is_resume_request(cfgs.MODEL.CKPT)

    # savedir
    if args.resume_dir:
        args.output_dir = os.path.abspath(args.resume_dir)
        if not os.path.isdir(args.output_dir):
            raise FileNotFoundError(f'--resume_dir not found: {args.output_dir}')
        if not resume_requested:
            raise ValueError(
                "Resuming requires MODEL.CKPT='best' (or epoch) / --ckpt best "
                f'(got CKPT={cfgs.MODEL.CKPT!r})'
            )
        print(f'Resuming in existing output directory: {args.output_dir}')
    else:
        args.output_dir = str(os.path.join(args.save_root_dir, args.exp_group_path, args.tag, args.extra_tag))

        # New run: if the folder already exists, append a timestamp (unless resuming).
        if os.path.exists(args.output_dir) and args.extra_tag != 'debug' and not resume_requested:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = f"{args.output_dir}_{timestamp}"
            print(f"Original output directory exists. Using new path: {args.output_dir}")
        elif resume_requested and not os.path.isdir(args.output_dir):
            raise FileNotFoundError(
                f'Resume CKPT={cfgs.MODEL.CKPT!r} but output_dir does not exist: {args.output_dir}. '
                f'Pass --resume_dir path/to/run (the folder that contains ckpt/).'
            )

    if args.dist_mode:
        dist.barrier()
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpt')
    if resume_requested:
        ckpt_path = common_utils.resolve_resume_ckpt_path(args.ckpt_dir, cfgs.MODEL.CKPT)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f'Resume checkpoint not found: {ckpt_path}\n'
                f'Available: {sorted(glob.glob(os.path.join(args.ckpt_dir, "checkpoint_*.pth")))}'
            )
    if not os.path.exists(args.ckpt_dir) and local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    if global_rank == 0:
        # Never allow training saves to land on the F5 / pretrained weight files.
        from openworld.f5_init import assert_not_overwriting_pretrained
        assert_not_overwriting_pretrained(
            os.path.join(args.ckpt_dir, 'checkpoint_best.pth'),
            [
                cfgs.MODEL.get('PRETRAINED_MODEL'),
                cfgs.MODEL.get('YOLO_PRETRAINED_MODEL'),
                cfgs.MODEL.get('ADAPTER_PRETRAINED_MODEL'),
            ],
        )

    if args.dist_mode:
        dist.barrier()

    # logger
    log_file = os.path.join(args.output_dir, 'train_{}_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), group_rank))
    logger = common_utils.create_logger(log_file, rank=local_rank)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard')) if global_rank == 0 else None
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_configs(cfgs, logger=logger)

    # trainer
    model_trainer = Trainer(args, cfgs, local_rank, global_rank, logger, tb_writer)

    tbar = tqdm.trange(model_trainer.last_epoch + 1, model_trainer.total_epochs,
                       desc='epochs', dynamic_ncols=True, disable=(local_rank != 0),
                       bar_format='{l_bar}{bar}{r_bar}\n')
    eval_interval = int(cfgs.TRAINER.get('EVAL_INTERVAL', 5))
    # train loop: eval every N epochs; save only when val best metric improves
    for current_epoch in tbar:
        model_trainer.train(current_epoch, tbar)
        should_eval = (
            ((current_epoch + 1) % eval_interval == 0)
            or (current_epoch == model_trainer.total_epochs - 1)
        )
        if should_eval:
            metrics = model_trainer.evaluate(current_epoch)
            model_trainer.save_best_ckpt(current_epoch, metrics)


if __name__ == '__main__':
    main()
