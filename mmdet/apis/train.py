# Copyright (C) 2018-2021 OpenMMLab
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import shutil
from os import path as osp
import warnings
import numpy as np
import random
import torch
from copy import copy
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner, LoggerHook,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer, load_checkpoint,
                         build_runner)

from mmdet.core import (DistEvalHook, DistEvalPlusBeforeRunHook, EvalHook,
                        EvalPlusBeforeRunHook)
from mmdet.integration.nncf import CompressionHook
from mmdet.integration.nncf import CheckpointHookBeforeTraining
from mmdet.integration.nncf import wrap_nncf_model
from mmdet.integration.nncf import AccuracyAwareLrUpdater
from mmdet.integration.nncf import is_accuracy_aware_training_set
from mmcv.utils import build_from_cfg

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger
from mmdet.utils import prepare_mmdet_model_for_execution
from .fake_input import get_fake_input


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def add_logging_on_first_and_last_iter(runner):
    def every_n_inner_iters(self, runner, n):
        if runner.inner_iter == 0 or runner.inner_iter == runner.max_iters - 1:
            return True
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    for hook in runner.hooks:
        if isinstance(hook, LoggerHook):
            hook.every_n_inner_iters = every_n_inner_iters.__get__(hook)


def build_val_dataloader(cfg, distributed):
    # Support batch_size > 1 in validation
    val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
    if val_samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.val.pipeline = replace_ImageToTensor(
            cfg.data.val.pipeline)
    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    val_dataloader = build_dataloader(
        val_dataset,
        samples_per_gpu=val_samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    return val_dataloader


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None,
                   val_dataloader=None,
                   compression_ctrl=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    map_location = 'cuda'
    if not torch.cuda.is_available():
        map_location = 'cpu'

    if cfg.load_from:
        load_checkpoint(model=model, filename=cfg.load_from, map_location=map_location)

    # put model on gpus
    if torch.cuda.is_available():
        model = model.cuda()

    if validate and not val_dataloader:
        val_dataloader = build_val_dataloader(cfg, distributed)

    # nncf model wrapper
    nncf_enable_compression = 'nncf_config' in cfg
    nncf_config = cfg.get('nncf_config', {})
    nncf_is_acc_aware_training_set = is_accuracy_aware_training_set(nncf_config)

    if not compression_ctrl and nncf_enable_compression:
        dataloader_for_init = data_loaders[0]
        compression_ctrl, model = wrap_nncf_model(model, cfg,
                                                  distributed=distributed,
                                                  val_dataloader=val_dataloader,
                                                  dataloader_for_init=dataloader_for_init,
                                                  get_fake_input_func=get_fake_input,
                                                  is_accuracy_aware=nncf_is_acc_aware_training_set)

    model = prepare_mmdet_model_for_execution(model, cfg, distributed)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    if nncf_is_acc_aware_training_set:
        # Prepare runner for Accuracy Aware
        cfg.runner = {
            'type': 'AccuracyAwareRunner',
            'target_metric_name': nncf_config['target_metric_name']
        }

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        grad_clip = cfg.optimizer_config.get('grad_clip', None)
        optimizer_config = Fp16OptimizerHook(
            **fp16_cfg, grad_clip=grad_clip, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(None, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    # register lr updater hook
    policy_type = cfg.lr_config.pop('policy')
    if policy_type == policy_type.lower():
        policy_type = policy_type.title()
    cfg.lr_config['type'] = policy_type + 'LrUpdaterHook'
    lr_updater_hook = build_from_cfg(cfg.lr_config, HOOKS)
    runner.register_lr_hook(lr_updater_hook)

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    add_logging_on_first_and_last_iter(runner)

    # register eval hooks
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        if nncf_enable_compression:
            # disable saving best snapshot, because it works incorrectly for NNCF,
            # best metric can be reached on not target compression rate.
            eval_cfg.pop('save_best')
            # enable evaluation after initialization of compressed model,
            # target accuracy can be reached without fine-tuning model
            eval_hook = DistEvalPlusBeforeRunHook if distributed else EvalPlusBeforeRunHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if nncf_enable_compression:
        runner.register_hook(CompressionHook(compression_ctrl=compression_ctrl))
        runner.register_hook(CheckpointHookBeforeTraining())
    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            if nncf_is_acc_aware_training_set and hook_cfg.get('type') == 'EarlyStoppingHook':
                continue
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from, map_location=map_location)
        if "best_ckpt" in runner.meta['hook_msgs']:
            runner.meta['hook_msgs']['best_ckpt'] = osp.join(cfg.work_dir,
                osp.basename(cfg.resume_from))
        shutil.copy(cfg.resume_from, cfg.work_dir)

    if nncf_is_acc_aware_training_set:
        def configure_optimizers_fn():
            optimizer = build_optimizer(runner.model, cfg.optimizer)
            lr_scheduler = AccuracyAwareLrUpdater(lr_updater_hook, runner, optimizer)
            return optimizer, lr_scheduler

        runner.run(data_loaders, cfg.workflow,
                   compression_ctrl=compression_ctrl,
                   configure_optimizers_fn=configure_optimizers_fn,
                   nncf_config=nncf_config)
    else:
        runner.run(data_loaders, cfg.workflow, compression_ctrl=compression_ctrl)
