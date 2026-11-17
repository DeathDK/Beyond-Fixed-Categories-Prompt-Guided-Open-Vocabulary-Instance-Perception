# @Time    : 2024/1/20 03:13
# @Author  : zhangchenming
import math
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist

from functools import partial
from stereo.datasets import build_dataloader
from stereo.utils import common_utils
from stereo.utils.common_utils import color_map_tensorboard, write_tensorboard
from stereo.utils.warmup import LinearWarmup
from stereo.utils.clip_grad import ClipGrad
from stereo.utils.lamb import Lamb
from stereo.utils.model_ema import ModelEMA
from stereo.evaluation.metric_per_image import epe_metric, d1_metric, threshold_metric


def build_warmup_cosine_scheduler(optimizer, total_units, warmup_units, eta_min):
    """Build a warmup-cosine scheduler in an explicitly selected time unit."""
    if total_units < 2:
        raise ValueError('total_units must be at least 2')
    warmup_units = max(1, min(int(warmup_units), int(total_units) - 1))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_units
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_units - warmup_units, eta_min=eta_min
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_units]
    )


def optimizer_group_gradient_norms(optimizer):
    """Return unscaled L2 gradient norms for named optimizer groups."""
    norms = {}
    for index, group in enumerate(optimizer.param_groups):
        name = group.get('name', f'group_{index}')
        gradients = [
            parameter.grad.detach().float()
            for parameter in group['params']
            if parameter.grad is not None
        ]
        if gradients:
            # Keep all per-parameter reductions on the device and synchronize
            # only once per optimizer group when producing the log scalar.
            total_norm = torch.nn.utils.get_total_norm(
                gradients,
                norm_type=2.0,
                foreach=True,
            )
            norm_value = total_norm.item()
        else:
            norm_value = 0.0
        norms[f'scalar/train/grad_norm_{name}'] = norm_value
    return norms


def assert_finite_training_state(loss, model, epoch, iteration, check_gradients=True):
    """Abort before an optimizer step when loss or gradients become non-finite."""
    if not torch.isfinite(loss.detach()).all().item():
        raise FloatingPointError(
            f'Non-finite loss at epoch={epoch}, iteration={iteration}: {loss.detach().item()}'
        )
    if not check_gradients:
        return
    parameters_with_grad = [
        parameter for parameter in model.parameters() if parameter.grad is not None
    ]
    try:
        torch.nn.utils.clip_grad_norm_(
            parameters_with_grad,
            max_norm=float('inf'),
            error_if_nonfinite=True,
            foreach=True,
        )
    except RuntimeError as error:
        raise FloatingPointError(
            f'Non-finite gradient norm at epoch={epoch}, iteration={iteration}'
        ) from error


def load_projection_adapter(model, filename, device, dist_mode=False):
    """Load and validate the three calibrated MobileNet-to-YOLO projections."""
    checkpoint = torch.load(filename, map_location=device)
    adapter_state = checkpoint.get('model_state', checkpoint)
    target_model = model.module if dist_mode else model
    target_state = target_model.state_dict()
    prefixes = ('proj_p3.', 'proj_p4.', 'proj_p5.')
    required = {key for key in target_state if key.startswith(prefixes)}
    provided = {key for key in adapter_state if key.startswith(prefixes)}
    missing = sorted(required - provided)
    mismatched = sorted(
        key for key in required & provided
        if target_state[key].shape != adapter_state[key].shape
    )
    if missing or mismatched:
        raise RuntimeError(
            f'Invalid projection adapter {filename}: '
            f'missing={missing}, shape_mismatch={mismatched}'
        )
    target_model.load_state_dict(
        {key: adapter_state[key] for key in required},
        strict=False,
    )
    return sorted(required)


def projection_adapter_required(model, loaded_checkpoint_keys, dist_mode=False):
    """Return whether the main checkpoint omitted any projection tensor."""
    target_model = model.module if dist_mode else model
    prefixes = ('proj_p3.', 'proj_p4.', 'proj_p5.')
    required = {key for key in target_model.state_dict() if key.startswith(prefixes)}
    return not required.issubset(set(loaded_checkpoint_keys))


class TrainerTemplate:
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer, model):
        self.args = args
        self.cfgs = cfgs
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.logger = logger
        self.tb_writer = tb_writer
        self.global_step = 0
        self.device = torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        )

        self.model = self.build_model(model)
        self.model_ema = None

        if self.args.run_mode in ['train', 'eval']:
            self.eval_set, self.eval_loader, self.eval_sampler = self.build_eval_loader()

        if self.args.run_mode == 'train':
            self.train_set, self.train_loader, self.train_sampler = self.build_train_loader()

            self.total_epochs = cfgs.OPTIMIZATION.NUM_EPOCHS
            self.last_epoch = -1
            # Independent best trackers: {metric_name: {'value': float, 'epoch': int}}
            self.best_trackers = {}
            # Legacy aliases (primary / first BEST_METRICS entry, usually epe)
            self.best_metric = None
            self.best_epoch = -1
            self.resume_from_best = False

            self.optimizer, self.scheduler = self.build_optimizer_and_scheduler()
            self.scaler = torch.cuda.amp.GradScaler(enabled=cfgs.OPTIMIZATION.AMP)
            ema_cfg = self.cfgs.OPTIMIZATION.get('EMA', {})
            if ema_cfg.get('ENABLED', False):
                self.model_ema = ModelEMA(
                    self.model,
                    decay=ema_cfg.get('DECAY', 0.9999),
                    tau=ema_cfg.get('TAU', 200.0),
                )
                self.logger.info(
                    'EMA enabled: decay=%.6f, tau=%.1f',
                    self.model_ema.decay_rate,
                    self.model_ema.tau,
                )

            if common_utils.is_resume_request(self.cfgs.MODEL.CKPT):
                self.resume_ckpt()

            self.warmup_scheduler = self.build_warmup()
            self.clip_gard = self.build_clip_grad()

    def build_train_loader(self):
        train_set, train_loader, train_sampler = build_dataloader(
            data_cfg=self.cfgs.DATA_CONFIG,
            batch_size=self.cfgs.OPTIMIZATION.BATCH_SIZE_PER_GPU,
            is_dist=self.args.dist_mode,
            workers=self.args.workers,
            pin_memory=self.args.pin_memory,
            mode='training')
        self.logger.info('Total samples for train dataset: %d' % (len(train_set)))
        if hasattr(train_sampler, 'summary'):
            self.logger.info(
                'Class-aware epoch size: %d (%d base + %d capped repeats)',
                train_sampler.epoch_size,
                len(train_set),
                train_sampler.extra_samples,
            )
            for class_id, stats in train_sampler.summary().items():
                self.logger.info(
                    'Class-aware sampling class=%s: frequency=%.4f, '
                    'expected=%.4f, repeat_factor=%.3f',
                    class_id,
                    stats['original_frequency'],
                    stats['expected_frequency'],
                    stats['repeat_factor'],
                )
        return train_set, train_loader, train_sampler

    def build_eval_loader(self):
        eval_set, eval_loader, eval_sampler = build_dataloader(
            data_cfg=self.cfgs.DATA_CONFIG,
            batch_size=self.cfgs.EVALUATOR.BATCH_SIZE_PER_GPU,
            is_dist=self.args.dist_mode,
            workers=self.args.workers,
            pin_memory=self.args.pin_memory,
            mode='evaluating')
        self.logger.info('Total samples for eval dataset: %d' % (len(eval_set)))
        return eval_set, eval_loader, eval_sampler

    def build_model(self, model):
        if self.cfgs.OPTIMIZATION.get('FREEZE_BN', False):
            model = common_utils.freeze_bn(model)
            self.logger.info('Freeze the batch normalization layers')

        if self.cfgs.OPTIMIZATION.SYNC_BN and self.args.dist_mode:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.logger.info('Convert batch norm to sync batch norm')
        model = model.to(self.device)

        if self.args.dist_mode:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=self.cfgs.MODEL.FIND_UNUSED_PARAMETERS)

        # load pretrained model (skip when resuming a training checkpoint)
        loaded_checkpoint_keys = set()
        resume_training = (
            self.args.run_mode == 'train'
            and common_utils.is_resume_request(self.cfgs.MODEL.get('CKPT', -1))
        )
        if self.cfgs.MODEL.PRETRAINED_MODEL and not resume_training:
            self.logger.info('Loading parameters from checkpoint %s' % self.cfgs.MODEL.PRETRAINED_MODEL)
            if not os.path.isfile(self.cfgs.MODEL.PRETRAINED_MODEL):
                raise FileNotFoundError
            # When F5_INIT is on, do not overlay myyolo.pt after the F5 checkpoint
            # (shape-matched tensors already come from F5; class head is remapped below).
            f5_init_cfg = self.cfgs.MODEL.get('F5_INIT', None)
            f5_init_enabled = bool(f5_init_cfg and f5_init_cfg.get('ENABLED', False))
            yolo_pretrained = self.cfgs.MODEL.get('YOLO_PRETRAINED_MODEL')
            if f5_init_enabled:
                yolo_pretrained = None
                self.logger.info(
                    'F5_INIT enabled: loading %s (read-only); YOLO_PRETRAINED_MODEL overlay disabled',
                    self.cfgs.MODEL.PRETRAINED_MODEL,
                )
            load_report = common_utils.load_params_from_file(
                model,
                self.cfgs.MODEL.PRETRAINED_MODEL,
                device=self.device,
                dist_mode=self.args.dist_mode,
                logger=self.logger,
                strict=False,
                yolo_pretrained_model=yolo_pretrained,
                state_key=self.cfgs.MODEL.get('CHECKPOINT_STATE', 'model_state'),
                yolo_load_class_head=bool(
                    self.cfgs.MODEL.get('YOLO_LOAD_CLASS_HEAD', True)
                ),
            )
            loaded_checkpoint_keys = load_report['updated_keys']
            if f5_init_enabled:
                from openworld.f5_init import (
                    apply_f5_class_head_init,
                    resolve_f5_state_dict,
                )
                state_key = self.cfgs.MODEL.get('CHECKPOINT_STATE', 'model_state')
                # Read-only load for remapping leftover class-logit tensors.
                f5_ckpt = torch.load(
                    self.cfgs.MODEL.PRETRAINED_MODEL,
                    map_location=self.device,
                )
                f5_state = resolve_f5_state_dict(f5_ckpt, state_key=state_key)
                tmp_model = model.module if self.args.dist_mode else model
                known = list(self.cfgs.MODEL.get('KNOWN_CLASSES', []))
                if not known:
                    raise ValueError('F5_INIT requires MODEL.KNOWN_CLASSES')
                source_classes = list(
                    f5_init_cfg.get('SOURCE_CLASSES', [])
                    or ['Car', 'Truck', 'Bus', 'Bicycle', 'Person']
                )
                apply_f5_class_head_init(
                    tmp_model,
                    f5_state,
                    known_classes=known,
                    source_classes=source_classes,
                    mode=str(f5_init_cfg.get('CLASS_HEAD_MODE', 'remap')),
                    logger=self.logger,
                )
                del f5_ckpt, f5_state
        elif resume_training:
            self.logger.info(
                'Skipping PRETRAINED_MODEL / YOLO init; will resume from CKPT=%s',
                self.cfgs.MODEL.CKPT,
            )
        adapter_pretrained_model = self.cfgs.MODEL.get('ADAPTER_PRETRAINED_MODEL')
        needs_projection_adapter = (
            (not resume_training)
            and projection_adapter_required(
                model,
                loaded_checkpoint_keys,
                dist_mode=self.args.dist_mode,
            )
        )
        if adapter_pretrained_model and needs_projection_adapter:
            self.logger.info(
                'Loading YOLO-aligned projection parameters from %s',
                adapter_pretrained_model,
            )
            if not os.path.isfile(adapter_pretrained_model):
                raise FileNotFoundError(
                    f'Adapter pretrained model not found: {adapter_pretrained_model}'
                )
            loaded_keys = load_projection_adapter(
                model,
                adapter_pretrained_model,
                self.device,
                dist_mode=self.args.dist_mode,
            )
            self.logger.info(
                'Loaded %d calibrated projection tensors', len(loaded_keys)
            )
        elif adapter_pretrained_model:
            self.logger.info(
                'Skipping projection adapter because the main checkpoint '
                'already contains all projection tensors'
            )
        return model

    def build_optimizer_and_scheduler(self):
        """Build transfer-aware parameter groups for joint fine-tuning."""
        if self.cfgs.OPTIMIZATION.get('USE_LEGACY_OPTIMIZER_GROUPS', False):
            return self._build_legacy_optimizer_and_scheduler()

        optimizer_config = self.cfgs.OPTIMIZATION.OPTIMIZER
        optimizer_cls = Lamb if optimizer_config.NAME == 'Lamb' else getattr(
            torch.optim, optimizer_config.NAME
        )
        base_lr = optimizer_config.LR
        defaults = {
            'backbone': 0.1,
            'projection': 1.0,
            'head_neck': 0.25,
            'head_box': 0.5,
            'head_class': 3.0,
            'head_mask': 0.5,
            'cost_agg': 0.5,
            'refine': 0.5,
            'seg_post': 1.0,
            'default': 1.0,
        }
        configured = optimizer_config.get('LR_MULTIPLIERS', {})
        multipliers = {
            name: float(configured.get(name, value))
            for name, value in defaults.items()
        }
        target_model = self.model.module if self.args.dist_mode else self.model
        processed = set()
        parameter_groups = []

        def add_group(name, modules):
            parameters = []
            for module in modules:
                if module is None:
                    continue
                for parameter in module.parameters():
                    if parameter.requires_grad and id(parameter) not in processed:
                        parameters.append(parameter)
                        processed.add(id(parameter))
            if parameters:
                parameter_groups.append({
                    'params': parameters,
                    'lr': base_lr * multipliers[name],
                    'name': name,
                })

        add_group('backbone', [target_model.backbone])
        add_group('projection', [
            target_model.proj_p3,
            target_model.proj_p4,
            target_model.proj_p5,
        ])
        head = target_model.head
        add_group('head_neck', [
            head.c3k2_13,
            head.c3k2_16,
            head.down_p3,
            head.c3k2_19,
            head.down_p4,
            head.c3k2_22,
        ])
        add_group('head_box', [head.seg.cv2, head.seg.dfl])
        add_group('head_class', [head.seg.cv3])
        add_group('head_mask', [head.seg.proto, head.seg.cv4])
        add_group('cost_agg', [getattr(target_model, 'cost_agg', None)])
        add_group('refine', [
            getattr(target_model, name, None)
            for name in ('refine_1', 'refine_2', 'refine_3', 'stem_2')
        ])
        add_group('seg_post', [getattr(target_model, 'seg_post', None)])

        remaining = [
            parameter
            for parameter in target_model.parameters()
            if parameter.requires_grad and id(parameter) not in processed
        ]
        if remaining:
            parameter_groups.append({
                'params': remaining,
                'lr': base_lr * multipliers['default'],
                'name': 'other',
            })

        for index, group in enumerate(parameter_groups):
            count = sum(parameter.numel() for parameter in group['params'])
            print(
                f"Group {index} ({group['name']}): {count} params, "
                f"LR={group['lr']:.2e}"
            )

        valid_arg = common_utils.get_valid_args(
            optimizer_cls, optimizer_config, ['name', 'lr']
        )
        optimizer = optimizer_cls(params=parameter_groups, **valid_arg)
        on_epoch = bool(self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH)
        units_per_epoch = 1 if on_epoch else len(self.train_loader)
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_units=self.total_epochs * units_per_epoch,
            warmup_units=(
                self.cfgs.OPTIMIZATION.SCHEDULER.get('WARMUP_EPOCHS', 5)
                * units_per_epoch
            ),
            eta_min=optimizer_config.get('MIN_LR', 1.0e-6),
        )
        return optimizer, scheduler

    def _build_legacy_optimizer_and_scheduler(self):
        if self.cfgs.OPTIMIZATION.OPTIMIZER.NAME == 'Lamb':
            optimizer_cls = Lamb
        else:
            optimizer_cls = getattr(torch.optim, self.cfgs.OPTIMIZATION.OPTIMIZER.NAME)
        
        # Get base learning rate config
        base_lr = self.cfgs.OPTIMIZATION.OPTIMIZER.LR
        
        # Learning-rate multipliers for different modules
        lr_multipliers = self.cfgs.OPTIMIZATION.OPTIMIZER.LR_MULTIPLIERS

        # Track already-processed parameters
        processed_params = set()
        params_group = []
        
        # Helper: add module parameters to a group
        def add_params_to_group(module, group_name, lr_mult_key):
            if module is None:
                return []
            params_list = []
            for param in module.parameters():
                if param.requires_grad and id(param) not in processed_params:
                    params_list.append(param)
                    processed_params.add(id(param))
            return params_list
        
        # 1. Backbone params (lower LR)
        backbone_params = add_params_to_group(self.model.backbone, 'backbone', 'backbone')
        if backbone_params:
            params_group.append({
                'params': backbone_params, 
                'lr': base_lr * lr_multipliers['backbone'], 
                'name': 'backbone'
            })
        
        # 2. Segmentation head params (higher LR)
        head_params = add_params_to_group(self.model.head, 'head', 'head')
        if head_params:
            params_group.append({
                'params': head_params, 
                'lr': base_lr * lr_multipliers['head'], 
                'name': 'head'
            })
        
        # 3. Cost aggregation module params (medium LR)
        cost_agg_params = add_params_to_group(getattr(self.model, 'cost_agg', None), 'cost_agg', 'cost_agg')
        if cost_agg_params:
            params_group.append({
                'params': cost_agg_params, 
                'lr': base_lr * lr_multipliers['cost_agg'], 
                'name': 'cost_agg'
            })
        
        # 4. Disparity refinement module params (medium LR)
        refine_modules = []
        for attr in ['refine_1', 'refine_2', 'refine_3', 'stem_2']:
            if hasattr(self.model, attr):
                refine_modules.append(getattr(self.model, attr))
        
        refine_params = []
        for module in refine_modules:
            for param in module.parameters():
                if param.requires_grad and id(param) not in processed_params:
                    refine_params.append(param)
                    processed_params.add(id(param))
        
        if refine_params:
            params_group.append({
                'params': refine_params, 
                'lr': base_lr * lr_multipliers['refine'], 
                'name': 'refine'
            })
        
        # 5. Segmentation post-process module params (higher LR)
        seg_post_params = add_params_to_group(getattr(self.model, 'seg_post', None), 'seg_post', 'seg_post')
        if seg_post_params:
            params_group.append({
                'params': seg_post_params, 
                'lr': base_lr * lr_multipliers['seg_post'], 
                'name': 'seg_post'
            })
        
        # 6. Remaining params (default LR)
        other_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and id(param) not in processed_params:
                other_params.append(param)
                processed_params.add(id(param))
        
        if other_params:
            params_group.append({
                'params': other_params, 
                'lr': base_lr * lr_multipliers['default'], 
                'name': 'other'
            })

            # Detailed debug info
        print("=== Parameter group details ===")
        for i, group in enumerate(params_group):
            num_params = sum(p.numel() for p in group['params'])
            lr = group['lr']
            wd = group.get('weight_decay', 'default')
            group_name = group.get('name', f'group_{i}')
            print(f"Group {i} ({group_name}): {num_params} params, LR={lr:.2e}, WD={wd}")
        
        # Create optimizer
        valid_arg = common_utils.get_valid_args(optimizer_cls, self.cfgs.OPTIMIZATION.OPTIMIZER, ['name', 'lr'])
        optimizer = optimizer_cls(params=params_group, **valid_arg)

        
        num_warmup_epochs = self.cfgs.OPTIMIZATION.SCHEDULER.get('WARMUP_EPOCHS', 5)
        eta_min = self.cfgs.OPTIMIZATION.OPTIMIZER.MIN_LR
        on_epoch = bool(self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH)
        units_per_epoch = 1 if on_epoch else len(self.train_loader)
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_units=self.total_epochs * units_per_epoch,
            warmup_units=num_warmup_epochs * units_per_epoch,
            eta_min=eta_min,
        )

        
        return optimizer, scheduler

    def resume_ckpt(self):
        ckpt_spec = self.cfgs.MODEL.CKPT
        spec_l = str(ckpt_spec).strip().lower() if isinstance(ckpt_spec, str) else ''
        self.resume_from_best = spec_l == 'best' or spec_l.startswith('best_')
        ckpt_path = common_utils.resolve_resume_ckpt_path(self.args.ckpt_dir, ckpt_spec)
        self.logger.info('Resume from %s (spec=%r)', ckpt_path, ckpt_spec)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f'Resume checkpoint not found: {ckpt_path}')
        try:
            checkpoint = torch.load(
                ckpt_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.last_epoch = int(checkpoint['epoch'])
        if checkpoint.get('best_trackers'):
            self.best_trackers = {
                str(k): {
                    'value': float(v['value']),
                    'epoch': int(v['epoch']),
                }
                for k, v in checkpoint['best_trackers'].items()
            }
        if checkpoint.get('best_metric', None) is not None:
            self.best_metric = float(checkpoint['best_metric'])
            self.best_epoch = int(checkpoint.get('best_epoch', self.last_epoch))
            # Seed legacy single-metric tracker if multi-best dict is empty.
            if not self.best_trackers:
                legacy_name = str(
                    checkpoint.get(
                        'best_metric_name',
                        self.cfgs.TRAINER.get('BEST_METRIC', 'epe'),
                    )
                )
                self.best_trackers[legacy_name] = {
                    'value': self.best_metric,
                    'epoch': self.best_epoch,
                }
        training_model_state = checkpoint.get('raw_model_state', checkpoint['model_state'])
        if self.args.dist_mode:
            self.model.module.load_state_dict(training_model_state)
        else:
            self.model.load_state_dict(training_model_state)
        if self.model_ema is not None:
            self.model_ema.load_state_dict(
                checkpoint.get('model_state', training_model_state),
                updates=checkpoint.get('ema_updates', 0),
            )
        if 'scaler_state' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state'])

        # Keep the same warmup/cosine recipe as a fresh run. Rebuild for the
        # configured NUM_EPOCHS, then advance the schedule.
        num_warmup_epochs = int(
            self.cfgs.OPTIMIZATION.SCHEDULER.get('WARMUP_EPOCHS', 5)
        )
        eta_min = self.cfgs.OPTIMIZATION.OPTIMIZER.MIN_LR
        on_epoch = bool(self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH)
        units_per_epoch = 1 if on_epoch else len(self.train_loader)
        self.scheduler = build_warmup_cosine_scheduler(
            self.optimizer,
            total_units=self.total_epochs * units_per_epoch,
            warmup_units=num_warmup_epochs * units_per_epoch,
            eta_min=eta_min,
        )
        # Align LR to the epoch where this checkpoint was saved (best or numbered).
        steps_done = self.last_epoch + 1
        for _ in range(steps_done):
            self.scheduler.step()
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        # Keep logging / TB step continuous after resume.
        self.global_step = (self.last_epoch + 1) * len(self.train_loader)
        self.logger.info(
            'Resumed %s epoch %d; next epoch %d / %d (LR schedule advanced %d steps)',
            'best' if self.resume_from_best else 'ckpt',
            self.last_epoch,
            self.last_epoch + 1,
            self.total_epochs,
            steps_done,
        )
    def build_warmup(self):
        last_step = (self.last_epoch + 1) * len(self.train_loader) - 1
        if 'WARMUP' in self.cfgs.OPTIMIZATION.SCHEDULER:
            warmup_steps = self.cfgs.OPTIMIZATION.SCHEDULER.WARMUP.get('WARM_STEPS', 1)
            warmup_scheduler = LinearWarmup(
                self.optimizer,
                warmup_period=warmup_steps,
                last_step=last_step)
        else:
            warmup_scheduler = LinearWarmup(
                self.optimizer,
                warmup_period=1,
                last_step=last_step)

        return warmup_scheduler

    def build_clip_grad(self):
        clip_gard = None
        if 'CLIP_GRAD' in self.cfgs.OPTIMIZATION:
            clip_type = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('TYPE', None)
            clip_value = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('CLIP_VALUE', 0.1)
            max_norm = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('MAX_NORM', 35)
            norm_type = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('NORM_TYPE', 2)
            clip_gard = ClipGrad(clip_type, clip_value, max_norm, norm_type)
        return clip_gard

    def train(self, current_epoch, tbar):
        self.model.train()
        train_model = self.model.module if self.args.dist_mode else self.model
        if hasattr(train_model, 'set_train_epoch'):
            train_model.set_train_epoch(current_epoch)
        if self.cfgs.OPTIMIZATION.get('FREEZE_BN', False):
            self.model = common_utils.freeze_bn(self.model)
        if hasattr(self.train_sampler, 'set_epoch'):
            self.train_sampler.set_epoch(current_epoch)
        self.train_one_epoch(current_epoch=current_epoch, tbar=tbar)
        if self.args.dist_mode:
            dist.barrier()
        if self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
            self.scheduler.step()
            self.warmup_scheduler.lrs = [group['lr'] for group in self.optimizer.param_groups]

    def evaluate(self, current_epoch):
        evaluation_model = self.model_ema.ema if self.model_ema is not None else self.model
        evaluation_model.eval()
        results = self.eval_one_epoch(current_epoch=current_epoch, model=evaluation_model)
        if self.args.dist_mode:
            dist.barrier()
        return results

    def _metric_specs(self):
        """Return list of (name, higher_is_better, filename).

        Prefer TRAINER.BEST_METRICS (independent multi-best). Fall back to legacy
        single TRAINER.BEST_METRIC + BEST_HIGHER_IS_BETTER.
        """
        higher_names = {'map', 'w_map', 'map50', 'w_map50', 'wmAP', 'w_mAP', 'mAP', 'mAP50', 'w_mAP50'}
        specs_cfg = self.cfgs.TRAINER.get('BEST_METRICS', None)
        specs = []
        if specs_cfg:
            for item in specs_cfg:
                if isinstance(item, str):
                    name = str(item)
                    higher = name in higher_names or name.lower() in {
                        'map', 'w_map', 'map50', 'w_map50'
                    }
                    fname = f'checkpoint_best_{name}.pth'
                else:
                    name = str(item.get('NAME', item.get('name')))
                    default_higher = name in higher_names or name.lower() in {
                        'map', 'w_map', 'map50', 'w_map50'
                    }
                    higher = bool(item.get('HIGHER_IS_BETTER', default_higher))
                    fname = str(
                        item.get('FILE', item.get('file', f'checkpoint_best_{name}.pth'))
                    )
                specs.append((name, higher, fname))
            return specs

        name = str(self.cfgs.TRAINER.get('BEST_METRIC', 'epe'))
        higher = bool(self.cfgs.TRAINER.get('BEST_HIGHER_IS_BETTER', False))
        return [(name, higher, 'checkpoint_best.pth')]

    @staticmethod
    def _metric_improved(value: float, best: float, higher_better: bool) -> bool:
        if best is None:
            return True
        return value > best if higher_better else value < best

    def _write_ckpt_file(self, filename, current_epoch, extra_state=None):
        from openworld.f5_init import assert_not_overwriting_pretrained
        assert_not_overwriting_pretrained(
            filename,
            [
                self.cfgs.MODEL.get('PRETRAINED_MODEL'),
                self.cfgs.MODEL.get('YOLO_PRETRAINED_MODEL'),
                self.cfgs.MODEL.get('ADAPTER_PRETRAINED_MODEL'),
            ],
        )
        state = {
            'best_trackers': {
                k: {'value': float(v['value']), 'epoch': int(v['epoch'])}
                for k, v in self.best_trackers.items()
            },
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
        }
        if extra_state:
            state.update(extra_state)
        common_utils.save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.args.dist_mode,
            current_epoch,
            filename=filename,
            model_ema=self.model_ema,
            extra_state=state,
        )

    def save_best_ckpt(self, current_epoch, metrics):
        """Save independent best checkpoints per metric (+ optional last ckpt).

        Each BEST_METRICS entry is tracked separately: epe improving does not
        require w_mAP to improve, and vice versa.
        """
        if metrics is None:
            return

        specs = self._metric_specs()
        save_last = bool(self.cfgs.TRAINER.get('SAVE_LAST_CKPT', True))

        if self.global_rank == 0:
            # Always refresh last checkpoint for interrupt-safe resume.
            if save_last:
                last_path = os.path.join(self.args.ckpt_dir, 'checkpoint_last.pth')
                self._write_ckpt_file(last_path, current_epoch)
                self.logger.info(
                    'Updated checkpoint_last.pth at epoch %d', current_epoch
                )

            for metric_name, higher_better, fname in specs:
                if metric_name not in metrics:
                    self.logger.warning(
                        'Skip best tracker %s: not in eval results %s',
                        metric_name,
                        list(metrics.keys()),
                    )
                    continue
                raw = metrics[metric_name]
                value = float(raw.item()) if torch.is_tensor(raw) else float(raw)
                if not math.isfinite(value):
                    self.logger.warning(
                        'Skip save %s: non-finite value %r at epoch %d',
                        metric_name,
                        value,
                        current_epoch,
                    )
                    continue

                tracker = self.best_trackers.get(metric_name)
                prev = None if tracker is None else float(tracker['value'])
                if not self._metric_improved(value, prev, higher_better):
                    self.logger.info(
                        'Skip save %s: %.6f not better than best=%.6f (epoch %d, %s)',
                        metric_name,
                        value,
                        prev,
                        int(tracker['epoch']),
                        'higher better' if higher_better else 'lower better',
                    )
                    continue

                self.best_trackers[metric_name] = {
                    'value': value,
                    'epoch': int(current_epoch),
                }
                # Keep legacy fields aligned with the first configured metric.
                if metric_name == specs[0][0]:
                    self.best_metric = value
                    self.best_epoch = int(current_epoch)

                ckpt_name = os.path.join(self.args.ckpt_dir, fname)
                self._write_ckpt_file(
                    ckpt_name,
                    current_epoch,
                    extra_state={
                        'best_metric_name': metric_name,
                        'best_higher_is_better': higher_better,
                        'best_metric': value,
                        'best_epoch': int(current_epoch),
                    },
                )
                # Resume compatibility: also mirror primary epe best as checkpoint_best.pth
                if metric_name.lower() == 'epe' and fname != 'checkpoint_best.pth':
                    alias = os.path.join(self.args.ckpt_dir, 'checkpoint_best.pth')
                    self._write_ckpt_file(
                        alias,
                        current_epoch,
                        extra_state={
                            'best_metric_name': metric_name,
                            'best_higher_is_better': higher_better,
                            'best_metric': value,
                            'best_epoch': int(current_epoch),
                        },
                    )
                self.logger.info(
                    'New best %s=%.6f (prev=%s) at epoch %d -> %s',
                    metric_name,
                    value,
                    'None' if prev is None else f'{prev:.6f}',
                    current_epoch,
                    ckpt_name,
                )

        if self.args.dist_mode:
            dist.barrier()

    def save_ckpt(self, current_epoch):
        """Deprecated epoch dumps; kept as no-op for callers. Use save_best_ckpt."""
        return

    def train_one_epoch(self, current_epoch, tbar):
        start_epoch = self.last_epoch + 1
        logger_iter_interval = self.cfgs.TRAINER.LOGGER_ITER_INTERVAL
        total_loss = torch.zeros((), device=self.device)
        loss_func = self.model.module.get_loss if self.args.dist_mode else self.model.get_loss

        # Initialize global_step if not already set
        if not hasattr(self, 'global_step'):
            self.global_step = 0
        
        train_loader_iter = iter(self.train_loader)
        for i in range(0, len(self.train_loader)):
            total_iter = current_epoch * len(self.train_loader) + i
            should_log = total_iter % logger_iter_interval == 0
            self.optimizer.zero_grad()
            lr = self.optimizer.param_groups[0]['lr']

            start_timer = time.time()
            data = next(train_loader_iter)
            for k, v in data.items():
                data[k] = v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
            data_timer = time.time()

            with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
                model_pred = self.model(data)
                infer_timer = time.time()
                loss, tb_info = loss_func(
                    model_pred,
                    data,
                    collect_metrics=should_log,
                )

            assert_finite_training_state(
                loss, self.model, current_epoch, i, check_gradients=False
            )

            # Do not call under autocast; calls backward() on scaled loss to create scaled gradients.
            self.scaler.scale(loss).backward()
            # Unscale before gradient clipping; unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            assert_finite_training_state(loss, self.model, current_epoch, i)
            if should_log:
                tb_info.update(optimizer_group_gradient_norms(self.optimizer))
            # Gradient clipping
            if self.clip_gard is not None:
                self.clip_gard(self.model)
            # optimizer's gradients are already unscaled, so scaler.step does not unscale them
            self.scaler.step(self.optimizer)
            # Updates the scale for next iteration.
            self.scaler.update()
            if self.model_ema is not None:
                self.model_ema.update(self.model)
            # torch.cuda.empty_cache()

            # Do not use warmup_scheduler with period>1 together with batch_scheduler
            with self.warmup_scheduler.dampening():
                if not self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
                    self.scheduler.step()

            total_loss += loss.detach()
            
            # Log learning-rate info
            if should_log:
                lr_info = {}
                for group_idx, param_group in enumerate(self.optimizer.param_groups):
                    group_name = param_group.get('name', f'group_{group_idx}')
                    lr_info[f'learning_rate/{group_name}'] = param_group['lr']
                
                # Write learning rates to TensorBoard
                if self.tb_writer is not None:
                    for key, value in lr_info.items():
                        self.tb_writer.add_scalar(key, value, self.global_step)
                
                # Also print to logger (optional)
                if self.logger is not None:
                    lr_details = ", ".join([f"{name.split('/')[-1]}:{lr:.2e}" for name, lr in lr_info.items()])
                    self.logger.info(f"Step {self.global_step} - Learning Rates: {lr_details}")

            trained_time_past_all = tbar.format_dict['elapsed']
            single_iter_second = trained_time_past_all / (total_iter + 1 - start_epoch * len(self.train_loader))
            remaining_second_all = single_iter_second * (self.total_epochs * len(self.train_loader) - total_iter - 1)
            
            if should_log:
                loss_value = tb_info['scalar/train/total_loss']
                average_loss = (total_loss / (i + 1)).item()
                message = ('Training Epoch:{:>2d}/{} Iter:{:>4d}/{} '
                        'Loss:{:#.6g}({:#.6g}) LR:{:.4e} '
                        'DataTime:{:.2f} InferTime:{:.2f}ms '
                        'Time cost: {}/{}'
                        ).format(current_epoch, self.total_epochs, i, len(self.train_loader),
                                    loss_value, average_loss, lr,
                                    data_timer - start_timer, (infer_timer - data_timer) * 1000,
                                    tbar.format_interval(trained_time_past_all),
                                    tbar.format_interval(remaining_second_all))
                self.logger.info(message)

            if self.cfgs.TRAINER.TRAIN_VISUALIZATION:
                tb_info['image/train/image'] = torch.cat([data['left'][0], data['right'][0]], dim=1) / 256
                tb_info['image/train/disp'] = color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])

            tb_info.update({'scalar/train/lr': lr})
            
            self.global_step += 1
            
            if should_log and self.local_rank == 0 and self.tb_writer is not None:
                write_tensorboard(self.tb_writer, tb_info, total_iter)

    @torch.no_grad()
    def eval_one_epoch(self, current_epoch, model=None):

        evaluation_model = self.model if model is None else model

        metric_func_dict = {
            'epe': epe_metric,
            'd1_all': d1_metric,
            'thres_1': partial(threshold_metric, threshold=1),
            'thres_2': partial(threshold_metric, threshold=2),
            'thres_3': partial(threshold_metric, threshold=3),
        }

        evaluator_cfgs = self.cfgs.EVALUATOR
        local_rank = self.local_rank
        stereo_metrics = [
            m for m in evaluator_cfgs.METRIC
            if m in metric_func_dict
        ]
        want_seg_ap = bool(
            self.cfgs.MODEL.get('DETECTION_ENABLED', False)
            and evaluator_cfgs.get('SEG_AP', True)
        )
        class_names = list(self.cfgs.MODEL.get('KNOWN_CLASSES', []) or [])
        if want_seg_ap and not class_names:
            self.logger.warning('SEG_AP enabled but MODEL.KNOWN_CLASSES is empty; skipping mAP.')
            want_seg_ap = False
        conf_threshold = float(evaluator_cfgs.get('SEG_AP_CONF', 0.001))
        pad_hw = tuple(evaluator_cfgs.get('SEG_AP_PAD_HW', [384, 1248]))

        epoch_metrics = {}
        for k in stereo_metrics:
            epoch_metrics[k] = {'indexes': [], 'values': []}
        seg_predictions = []
        seg_ground_truths = {}

        for i, data in enumerate(self.eval_loader):
            for k, v in data.items():
                data[k] = v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v

            with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
                infer_start = time.time()
                model_pred = evaluation_model(data)
                infer_time = time.time() - infer_start

            disp_pred = model_pred['disp_pred']
            disp_gt = data["disp"]
            mask = (disp_gt < evaluator_cfgs.MAX_DISP) & (disp_gt > 0)
            if 'occ_mask' in data and evaluator_cfgs.get('APPLY_OCC_MASK', False):
                mask = mask & ~data['occ_mask'].to(torch.bool)

            for m in stereo_metrics:
                metric_func = metric_func_dict[m]
                res = metric_func(disp_pred.squeeze(1), disp_gt, mask)
                epoch_metrics[m]['indexes'].extend(data['index'].tolist())
                epoch_metrics[m]['values'].extend(res.tolist())

            if want_seg_ap and 'seg_pred' in model_pred:
                from stereo.evaluation.seg_val_ap import (
                    build_gt_entry_from_seg_labels,
                    build_pred_entry_from_model,
                )
                batch = int(data['left'].shape[0])
                orig_sizes = data.get('orig_size', [None] * batch)
                names = data.get('name', [f'idx_{x}' for x in data['index'].tolist()])
                seg_labels_batch = data.get('seg_labels', [[] for _ in range(batch)])
                for b in range(batch):
                    image_id = str(names[b]) if not isinstance(names[b], str) else names[b]
                    image_id = os.path.basename(image_id)
                    orig_hw = orig_sizes[b]
                    if orig_hw is None:
                        # Fallback: assume content is bottom-left after RightTopPad.
                        th, tw = int(data['left'].shape[-2]), int(data['left'].shape[-1])
                        orig_hw = (th, tw)
                    else:
                        orig_hw = (int(orig_hw[0]), int(orig_hw[1]))
                    pred_entry = build_pred_entry_from_model(
                        model_pred,
                        class_names=class_names,
                        image_id=image_id,
                        orig_hw=orig_hw,
                        pad_hw=pad_hw,
                        conf_threshold=conf_threshold,
                        batch_index=b,
                    )
                    gt_entry = build_gt_entry_from_seg_labels(
                        seg_labels_batch[b],
                        image_id=image_id,
                        orig_hw=orig_hw,
                        pad_hw=pad_hw,
                    )
                    seg_predictions.append(pred_entry)
                    seg_ground_truths[image_id] = gt_entry

            if i % self.cfgs.TRAINER.LOGGER_ITER_INTERVAL == 0:
                message = ('Evaluating Epoch:{:>2d} Iter:{:>4d}/{} InferTime: {:.2f}ms'
                           ).format(current_epoch, i, len(self.eval_loader), infer_time * 1000)
                self.logger.info(message)

                if self.cfgs.TRAINER.EVAL_VISUALIZATION and self.tb_writer is not None:
                    tb_info = {
                        'image/eval/image': torch.cat([data['left'][0], data['right'][0]], dim=1) / 256,
                        'image/eval/disp': color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])
                    }
                    write_tensorboard(self.tb_writer, tb_info, current_epoch * len(self.eval_loader) + i)

        # gather from all gpus
        if self.args.dist_mode:
            dist.barrier()
            self.logger.info("Start reduce metrics.")
            for k in epoch_metrics.keys():
                indexes = torch.tensor(epoch_metrics[k]["indexes"]).to(local_rank)
                values = torch.tensor(epoch_metrics[k]["values"]).to(local_rank)
                gathered_indexes = [torch.zeros_like(indexes) for _ in range(dist.get_world_size())]
                gathered_values = [torch.zeros_like(values) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_indexes, indexes)
                dist.all_gather(gathered_values, values)
                unique_dict = {}
                for key, value in zip(torch.cat(gathered_indexes, dim=0).tolist(),
                                      torch.cat(gathered_values, dim=0).tolist()):
                    if key not in unique_dict:
                        unique_dict[key] = value
                epoch_metrics[k]["indexes"] = list(unique_dict.keys())
                epoch_metrics[k]["values"] = list(unique_dict.values())
            if want_seg_ap:
                gathered_preds = [None for _ in range(dist.get_world_size())]
                gathered_gts = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_preds, seg_predictions)
                dist.all_gather_object(gathered_gts, seg_ground_truths)
                if local_rank == 0:
                    merged_preds, merged_gts = [], {}
                    for part in gathered_preds:
                        merged_preds.extend(part or [])
                    for part in gathered_gts:
                        merged_gts.update(part or {})
                    seg_predictions, seg_ground_truths = merged_preds, merged_gts

        results = {}
        for k in epoch_metrics.keys():
            results[k] = torch.tensor(epoch_metrics[k]["values"]).mean()

        if want_seg_ap and (not self.args.dist_mode or local_rank == 0):
            from stereo.evaluation.seg_val_ap import compute_known_mask_ap
            try:
                seg_scores = compute_known_mask_ap(
                    seg_predictions, seg_ground_truths, class_names
                )
                results['mAP'] = torch.tensor(float(seg_scores.get('mAP', 0.0)))
                results['w_mAP'] = torch.tensor(float(seg_scores.get('w_mAP', 0.0)))
                results['mAP50'] = torch.tensor(float(seg_scores.get('mAP@.50', 0.0)))
                results['w_mAP50'] = torch.tensor(float(seg_scores.get('all_mAP@.50', 0.0)))
            except Exception as exc:
                self.logger.exception('SEG_AP failed: %s', exc)

        if self.args.dist_mode and want_seg_ap:
            # Broadcast scalar seg metrics so all ranks share identical results dict keys.
            def _scalar(key):
                if key not in results:
                    return float('nan')
                val = results[key]
                return float(val.item()) if torch.is_tensor(val) else float(val)

            payload = torch.tensor(
                [_scalar('mAP'), _scalar('w_mAP'), _scalar('mAP50'), _scalar('w_mAP50')],
                device=self.device,
            )
            dist.broadcast(payload, src=0)
            results['mAP'] = payload[0].detach().cpu()
            results['w_mAP'] = payload[1].detach().cpu()
            results['mAP50'] = payload[2].detach().cpu()
            results['w_mAP50'] = payload[3].detach().cpu()

        if local_rank == 0 and self.tb_writer is not None:
            tb_info = {}
            for k, v in results.items():
                tb_info[f'scalar/val/{k}'] = v.item() if torch.is_tensor(v) else float(v)

            write_tensorboard(self.tb_writer, tb_info, current_epoch)

        # Compact log line: stereo + segmentation.
        if local_rank == 0:
            stereo_msg = {k: results[k] for k in stereo_metrics if k in results}
            seg_msg = {
                k: results[k]
                for k in ('mAP', 'w_mAP', 'mAP50', 'w_mAP50')
                if k in results
            }
            self.logger.info(f"Epoch {current_epoch} metrics: {stereo_msg}")
            if seg_msg:
                self.logger.info(
                    "Epoch %d seg AP (known-only): mAP=%.4f w_mAP=%.4f mAP50=%.4f w_mAP50=%.4f",
                    current_epoch,
                    float(seg_msg.get('mAP', 0.0)),
                    float(seg_msg.get('w_mAP', 0.0)),
                    float(seg_msg.get('mAP50', 0.0)),
                    float(seg_msg.get('w_mAP50', 0.0)),
                )
        else:
            self.logger.info(f"Epoch {current_epoch} metrics: {results}")
        return results
