import os
import sys
import json
import argparse
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from robo3d.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from robo3d.utils.save import ModelSaver, save_training_meta
from robo3d.utils.misc import NoOp, set_dropout, set_random_seed
from robo3d.utils.distributed import set_cuda, wrap_model, all_gather

from robo3d.optim import get_lr_sched, get_lr_sched_decay_rate
from robo3d.optim.misc import build_optimizer

from robo3d.configs.rlbench.default import get_config

from robo3d.datasets.loader import build_dataloader
from robo3d.datasets.rlbench.keystep_dataset import (
    KeystepDataset, stepwise_collate_fn, episode_collate_fn
)
from robo3d.datasets.rlbench.pcd_keystep_dataset import (
    PCDKeystepDataset, ProcessedPCDKeystepDataset,
    pcd_stepwise_collate_fn, pcd_episode_collate_fn,
)

from robo3d.models.pct_manipulator import PCTManipulator

from robo3d.utils.slurm_requeue import init_signal_handler


dataset_factory = {
    'keystep_stepwise': (KeystepDataset, stepwise_collate_fn),
    'keystep_episode': (KeystepDataset, episode_collate_fn),
    'pre_pcd_keystep_stepwise': (ProcessedPCDKeystepDataset, pcd_stepwise_collate_fn),
    'pcd_keystep_stepwise': (PCDKeystepDataset, pcd_stepwise_collate_fn),
    'pre_pcd_keystep_episode': (ProcessedPCDKeystepDataset, pcd_episode_collate_fn),
    'pcd_keystep_episode': (PCDKeystepDataset, pcd_episode_collate_fn),
}
model_factory = {
    'PCTManipulator': PCTManipulator,
}


def main(config):
    config.defrost()
    default_gpu, n_gpu, device = set_cuda(config)

    if default_gpu:
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}'.format(
                device, n_gpu, bool(config.local_rank != -1)
            )
        )

    seed = config.SEED
    if config.local_rank != -1:
        seed += config.rank
    set_random_seed(seed)

    # load data training set
    dataset_class, dataset_collate_fn = dataset_factory[config.DATASET.dataset_class]

    dataset = dataset_class(**config.DATASET)
    data_loader, pre_epoch = build_dataloader(
        dataset, dataset_collate_fn, True, config
    )
    LOGGER.info(f'#num_steps_per_epoch: {len(data_loader)}')
    if config.TRAIN.num_train_steps is None:
        config.TRAIN.num_train_steps = len(data_loader) * config.TRAIN.num_epochs
    else:
        if config.TRAIN.num_epochs is not None:
            LOGGER.info('please do not set num_train_steps=%d and num_epochs=%d at the same time.' % (
                config.TRAIN.num_train_steps, config.TRAIN.num_epochs
            ))
        config.TRAIN.num_epochs = int(
            np.ceil(config.TRAIN.num_train_steps / len(data_loader)))
        
    if config.TRAIN.gradient_accumulation_steps > 1:
        config.TRAIN.num_train_steps *= config.TRAIN.gradient_accumulation_steps
        config.TRAIN.num_epochs *= config.TRAIN.gradient_accumulation_steps

    # setup loggers
    if default_gpu:
        save_training_meta(config)
        # TB_LOGGER.create(os.path.join(config.output_dir, 'logs'))
        if config.tfboard_log_dir is None:
            output_dir_tokens = config.output_dir.split('/')
            config.tfboard_log_dir = os.path.join(output_dir_tokens[0], 'TFBoard', *output_dir_tokens[1:])
        TB_LOGGER.create(config.tfboard_log_dir)
        model_saver = ModelSaver(os.path.join(config.output_dir, 'ckpts'))
        add_log_to_file(os.path.join(config.output_dir, 'logs', 'log.txt'))
    else:
        LOGGER.disabled = True
        model_saver = NoOp()

    # Prepare model
    model_class = model_factory[config.MODEL.model_class]
    model = model_class(**config.MODEL)
    # DDP: SyncBN
    if config.world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Fix parameters
    if config.TRAIN.freeze_params.encoder:
        for param_name, param in model.named_parameters():
            if param_name.startswith('mae_encoder') and 'decoder_block' not in param_name:
                    param.requires_grad = False
    if config.TRAIN.freeze_params.decoder:
        for param_name, param in model.named_parameters():
            if param_name.startswith('mae_encoder') and 'decoder_block' in param_name:
                param.requires_grad = False

    LOGGER.info("Model: nweights %d nparams %d" % (model.num_parameters))
    LOGGER.info("Model: trainable nweights %d nparams %d" %
                (model.num_trainable_parameters))
    
    config.freeze()

    # Load from checkpoint
    model_checkpoint_file = config.checkpoint
    optimizer_checkpoint_file = os.path.join(
        config.output_dir, 'ckpts', 'train_state_latest.pt'
    )
    if os.path.exists(optimizer_checkpoint_file) and config.TRAIN.resume_training:
        LOGGER.info('Load the optimizer checkpoint from %s' % optimizer_checkpoint_file)
        optimizer_checkpoint = torch.load(
            optimizer_checkpoint_file, map_location=lambda storage, loc: storage
        )
        lastest_model_checkpoint_file = os.path.join(
            config.output_dir, 'ckpts', 'model_step_%d.pt' % optimizer_checkpoint['step']
        )
        if os.path.exists(lastest_model_checkpoint_file):
            LOGGER.info('Load the model checkpoint from %s' % lastest_model_checkpoint_file)
            model_checkpoint_file = lastest_model_checkpoint_file
        global_step = optimizer_checkpoint['step']
        restart_epoch = global_step // len(data_loader)
    else:
        optimizer_checkpoint = None
        # to compute training statistics
        restart_epoch = 0
        global_step = restart_epoch * len(data_loader) 

    if model_checkpoint_file is not None:
        checkpoint = torch.load(
            model_checkpoint_file, map_location=lambda storage, loc: storage)
        LOGGER.info('Load the model checkpoint (%d params)' % len(checkpoint))
        new_checkpoint = {}
        state_dict = model.state_dict()
        for k, v in checkpoint.items():
            if (k in state_dict) and (state_dict[k].size() == v.size()):
                if config.TRAIN.resume_encoder_only and (k.startswith('mae_decoder') or 'decoder_block' in k):
                    continue
                new_checkpoint[k] = v
        LOGGER.info('Resumed the model checkpoint (%d params)' % len(new_checkpoint))
        model.load_state_dict(new_checkpoint, strict=config.checkpoint_strict_load)

    model.train()
    # set_dropout(model, config.TRAIN.dropout)
    model = wrap_model(model, device, config.local_rank, find_unused_parameters=True)

    # Prepare optimizer
    optimizer, init_lrs = build_optimizer(model, config.TRAIN)
    if optimizer_checkpoint is not None:
        optimizer.load_state_dict(optimizer_checkpoint['optimizer'])

    if default_gpu:
        pbar = tqdm(initial=global_step, total=config.TRAIN.num_train_steps)
    else:
        pbar = NoOp()

    LOGGER.info(f"***** Running training with {config.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d", config.TRAIN.train_batch_size if config.local_rank == -1 
                else config.TRAIN.train_batch_size * config.world_size)
    LOGGER.info("  Accumulate steps = %d", config.TRAIN.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", config.TRAIN.num_train_steps)

    optimizer.zero_grad()
    optimizer.step()

    init_signal_handler()

    running_metrics = {}

    for epoch_id in range(restart_epoch, config.TRAIN.num_epochs):
        if global_step >= config.TRAIN.num_train_steps:
            break

        # In distributed mode, calling the set_epoch() method at the beginning of each epoch
        pre_epoch(epoch_id)
        
        for step, batch in enumerate(data_loader):
            # forward pass
            losses, logits = model(batch, compute_loss=True)

            # backward pass
            if config.TRAIN.gradient_accumulation_steps > 1:  # average loss
                losses['total'] = losses['total'] / config.TRAIN.gradient_accumulation_steps
            losses['total'].backward()

            # compute accuracy for openness state
            acc = ((logits[..., -1].data.cpu() > 0)
                   == batch['actions'][..., -1].cpu()).float()
            if 'step_masks' in batch:
                acc = torch.sum(acc * batch['step_masks']) / \
                      torch.sum(batch['step_masks']).cpu()
            else:
                acc = acc.mean().cpu()

            for key, value in losses.items():
                TB_LOGGER.add_scalar(f'step/loss_{key}', value.item(), global_step)
                running_metrics.setdefault(f'loss_{key}', RunningMeter(f'loss_{key}'))
                running_metrics[f'loss_{key}'](value.item())
            TB_LOGGER.add_scalar('step/acc_open', acc.item(), global_step)
            running_metrics.setdefault('acc_open', RunningMeter('acc_open'))
            running_metrics['acc_open'](acc.item())

            # optimizer update and logging
            if (step + 1) % config.TRAIN.gradient_accumulation_steps == 0:
                global_step += 1
                # learning rate scheduling
                lr_decay_rate = get_lr_sched_decay_rate(global_step, config.TRAIN)
                for kp, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = lr_this_step = max(init_lrs[kp] * lr_decay_rate, 1e-8)
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                TB_LOGGER.step()

                # update model params
                if config.TRAIN.grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.grad_norm
                    )
                    # print(step, name, grad_norm)
                    # for k, v in model.named_parameters():
                    #     if v.grad is not None:
                    #         v = torch.norm(v).data.item()
                    #         print(k, v)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

            if global_step % config.TRAIN.log_steps == 0:
                # monitor training throughput
                LOGGER.info(
                    f'==============Epoch {epoch_id} Step {global_step}===============')
                LOGGER.info(', '.join(['%s:%.4f' % (lk, lv.val) for lk, lv in running_metrics.items()]))
                LOGGER.info('===============================================')

            if global_step % config.TRAIN.save_steps == 0:
                model_saver.save(model, global_step, optimizer=optimizer, rewrite_optimizer=True)

            if global_step >= config.TRAIN.num_train_steps:
                break

    if global_step % config.TRAIN.save_steps != 0:
        LOGGER.info(
            f'==============Epoch {epoch_id} Step {global_step}===============')
        LOGGER.info(', '.join(['%s:%.4f' % (lk, lv.val) for lk, lv in running_metrics.items()]))
        LOGGER.info('===============================================')
        model_saver.save(model, global_step, optimizer=optimizer, rewrite_optimizer=True)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line (use , to separate values in a list)",
    )
    args = parser.parse_args()

    config = get_config(args.exp_config, args.opts)

    if os.path.exists(config.output_dir) and os.listdir(config.output_dir):
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                config.output_dir
            )
        )

    return config


if __name__ == '__main__':
    config = build_args()
    main(config)
