import torch
from functools import partial
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .dataset_template import custom_collate

from .kitti_dataset import KittiDataset
from .samplers import CappedClassAwareSampler

__all__ = {
    'KittiDataset': KittiDataset,
}


def build_dataloader(data_cfg, batch_size, is_dist, workers, pin_memory, mode):
    all_dataset = []
    for data_info in data_cfg.DATA_INFOS:
        dataset = __all__[data_info.DATASET](
            data_info=data_info,
            data_cfg=data_cfg,
            mode=mode)
        all_dataset.append(dataset)
    dataset = torch.utils.data.ConcatDataset(all_dataset)

    # visualize_stereo_sample(dataset[0], save_path=None)

    shuffle = True if mode=='training' else False
    sampling_cfg = data_cfg.get('CLASS_AWARE_SAMPLING', {})
    use_class_aware_sampling = shuffle and sampling_cfg.get('ENABLED', False)
    if use_class_aware_sampling:
        sampler = CappedClassAwareSampler(
            dataset,
            repeat_threshold=sampling_cfg.get('REPEAT_THRESHOLD', 0.1),
            max_repeat_factor=sampling_cfg.get('MAX_REPEAT_FACTOR', 3.0),
            max_extra_fraction=sampling_cfg.get('MAX_EXTRA_FRACTION', 0.1),
            seed=sampling_cfg.get('SEED', 0),
        )
    elif is_dist:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    batch_uniform = data_cfg.DATA_TRANSFORM.get('BATCH_UNIFORM', False)
    batch_uniform = batch_uniform if mode=='training' else False
    random_type = data_cfg.DATA_TRANSFORM.get('RANDOM_TYPE', False)
    h_range = data_cfg.DATA_TRANSFORM.get('H_RANGE', False)
    w_range = data_cfg.DATA_TRANSFORM.get('W_RANGE', False)

    partial_custom_collate = partial(custom_collate, concat_dataset=dataset,
                                     batch_uniform=batch_uniform, random_type=random_type,
                                     h_range=h_range, w_range=w_range)

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        collate_fn=partial_custom_collate,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if workers > 0:
        loader_kwargs.update(
            prefetch_factor=2,       # each worker prefetches 2 batches
            persistent_workers=True  # avoid re-initializing workers
        )

    loader = DataLoader(**loader_kwargs)

    return dataset, loader, sampler
