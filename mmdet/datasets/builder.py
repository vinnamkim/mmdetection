# Copyright (C) 2018-2021 OpenMMLab
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import copy
import glob
import numpy as np
import platform
import random
from functools import partial
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader

from .samplers import DistributedGroupSampler, DistributedSampler, GroupSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')

from .coco import CocoDataset, ConcatenatedCocoDataset
from .coco_with_text import CocoWithTextDataset, ConcatenatedCocoWithTextDataset


def get_image_prefixes_auto(cfg, ann_files):
    del cfg['img_prefix_auto']
    assert cfg.get('img_prefix', None) is None
    if cfg['type'] == 'CustomCocoDataset':
        # assuming following dataset structure:
        # dataset_root
        # ├── annotations
        # │   ├── instances_train.json
        # │   ├── ...
        # ├── images
        #     ├── image_name1
        #     ├── image_name2
        #     ├── ...
        # and file_name inside instances_train.json is relative to <dataset_root>/images
        img_prefixes = \
            [os.path.join(os.path.dirname(ann_file), '..', 'images') for ann_file in ann_files]
    else:
        raise NotImplementedError

    return img_prefixes


def _concat_dataset(cfg, default_args=None):
    from .dataset_wrappers import ConcatDataset
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)
    separate_eval = cfg.get('separate_eval', True)

    if cfg.get('img_prefix_auto', False):
        img_prefixes = get_image_prefixes_auto(cfg, ann_files)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        # pop 'separate_eval' since it is not a valid key for common datasets.
        if 'separate_eval' in data_cfg:
            data_cfg.pop('separate_eval')
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    concatenated_dataset = ConcatDataset(datasets)
    if all(isinstance(dataset, CocoWithTextDataset) for dataset in concatenated_dataset.datasets):
        concatenated_dataset = ConcatenatedCocoWithTextDataset(concatenated_dataset)
    elif all(isinstance(dataset, CocoDataset) for dataset in concatenated_dataset.datasets):
        concatenated_dataset = ConcatenatedCocoDataset(concatenated_dataset)
    return concatenated_dataset


def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import (ConcatDataset, RepeatDataset,
                                   ClassBalancedDataset, MultiImageMixDataset,
                                   ImageTilingDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'MultiImageMixDataset':
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        cp_cfg.pop('labels')
        dataset = MultiImageMixDataset(**cp_cfg)
    elif cfg['type'] == 'ImageTilingDataset':
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        cp_cfg.pop('labels')
        dataset = ImageTilingDataset(**cp_cfg)
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        if 'ann_file' in cfg:
            # FIXME. Is it still of use?
            matches = glob.glob(cfg['ann_file'], recursive=True)
            if not matches:
                raise RuntimeError(f'Failed to find annotation files that match pattern: '
                                   f'{cfg["ann_file"]}')
            cfg['ann_file'] = matches
            if len(cfg['ann_file']) == 1:
                if cfg.get('img_prefix_auto', False):
                    cfg['img_prefix'] = get_image_prefixes_auto(cfg, cfg['ann_file'])[0]
                cfg['ann_file'] = cfg['ann_file'][0]
                dataset = build_from_cfg(cfg, DATASETS, default_args)
            else:
                dataset = _concat_dataset(cfg, default_args)
        else:
            dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = DistributedGroupSampler(dataset, samples_per_gpu,
                                              world_size, rank)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
