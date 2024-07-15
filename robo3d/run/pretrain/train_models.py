import os
import sys
import json
import argparse
import time
from collections import defaultdict
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# import numpy after torch: https://github.com/pytorch/pytorch/issues/101188
import numpy as np

from robo3d.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from robo3d.utils.save import ModelSaver, save_training_meta
from robo3d.utils.misc import NoOp, set_dropout, set_random_seed
from robo3d.utils.distributed import set_cuda, wrap_model, all_gather

from robo3d.optim import get_lr_sched, get_lr_sched_decay_rate
from robo3d.optim.misc import build_optimizer

from robo3d.configs.default import get_config

from robo3d.datasets.loader import (
    build_dataloader, MetaLoader, PrefetchLoader,
)
from robo3d.datasets.pointcloud.pretrain_dataset import (
    SingleObjectPretrainDataset, singleobj_pretrain_collate_fn
)
from robo3d.datasets.pointcloud.pretrain_multiobj_dataset import (
    MultiObjectPretrainDataset, multiobj_pretrain_collate_fn
)
from robo3d.datasets.pointcloud.acronym_grasp_dataset import (
    ObjectGraspDataset, grasp_pretrain_collate_fn
)
from robo3d.datasets.pointcloud.modelnet_dataset import (
    ModelNetDataset, modelnet_collate_fn
)
from robo3d.datasets.pointcloud.scanobjectnn_dataset import (
    ScanObjectNNDataset, scanobjectnn_collate_fn
)
from robo3d.datasets.pointcloud.objaverse_lvis_dataset import (
    ObjaverseLVISDataset, objaverse_lvis_collate_fn
)

from robo3d.models.pct_pretrain import PCPretrainModel
from robo3d.models.clip_encoder import ClipEncoder, OpenClipEncoder

from robo3d.utils.slurm_requeue import init_signal_handler


VAL_DATA_FACTORY = {
    'modelnet': (ModelNetDataset, modelnet_collate_fn),
    'scanobjectnn': (ScanObjectNNDataset, scanobjectnn_collate_fn),
    'objaverse_lvis': (ObjaverseLVISDataset, objaverse_lvis_collate_fn)
}
TRN_DATA_FACTORY = {
    'single_object': (SingleObjectPretrainDataset, singleobj_pretrain_collate_fn),
    'multi_objects': (MultiObjectPretrainDataset, multiobj_pretrain_collate_fn), 
    'grasp': (ObjectGraspDataset, grasp_pretrain_collate_fn),
}
MODEL_FACTORY = {
    'PCPretrainModel': PCPretrainModel
}

def create_dataloaders(config):
    task_dataset_configs = {
        'single_object': config.SINGLEOBJ_DATASET,
        'multi_objects': config.MULTIOBJ_DATASET,
        'grasp': config.GRASP_DATASET,
    }

    dataloaders = {}
    for task_name, task_ratio, task_batch_size in zip(
        config.TRAIN.trn_task_names, config.TRAIN.trn_task_ratios, config.TRAIN.trn_task_batch_size
    ):
        if task_ratio > 0:
            task_dataset = TRN_DATA_FACTORY[task_name][0](**task_dataset_configs[task_name])
            task_collate_fn = TRN_DATA_FACTORY[task_name][1]
            task_loader, pre_epoch = build_dataloader(
                task_dataset, task_collate_fn, True, config, batch_size=task_batch_size
            )
            dataloaders[task_name] = (task_loader, task_ratio, pre_epoch)
            LOGGER.info(f'train {task_name}: #num_data {len(task_dataset)}, #num_steps_per_epoch {len(task_loader)}')
    return dataloaders
        
def main(config):
    # torch.autograd.set_detect_anomaly(True)

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
    trn_dataloaders = create_dataloaders(config)
    meta_loader = MetaLoader(
        trn_dataloaders, accum_steps=config.TRAIN.gradient_accumulation_steps, 
        distributed=config.local_rank != -1, device=device
    )
    meta_loader = PrefetchLoader(meta_loader, device)

    # load data validation set
    val_dataloaders = {}
    for val_dataname in config.VAL_DATASET.dataset_names:
        val_dataset_class, val_dataset_collate_fn = VAL_DATA_FACTORY[val_dataname]
        val_dataset = val_dataset_class(**config.VAL_DATASET.dataset_cfgs[val_dataname])
        val_dataloader, _ = build_dataloader(
            val_dataset, val_dataset_collate_fn, False, config
        )
        LOGGER.info(f'val {val_dataname}: #num_data {len(val_dataset)}, #num_catorgies {val_dataset.num_categories}')
        val_dataloaders[val_dataname] = val_dataloader

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
    model = MODEL_FACTORY[config.MODEL.model_class](config.MODEL)
    # DDP: SyncBN
    if config.world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    LOGGER.info("Model: nweights %d nparams %d" % (model.num_parameters))
    LOGGER.info("Model: trainable nweights %d nparams %d" % (model.num_trainable_parameters))
    
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
    else:
        optimizer_checkpoint = None
        # to compute training statistics
        global_step = 0

    if model_checkpoint_file is not None:
        checkpoint = torch.load(
            model_checkpoint_file, map_location=lambda storage, loc: storage)
        LOGGER.info('Load the model checkpoint (%d params)' % len(checkpoint))
        new_checkpoint = {}
        state_dict = model.state_dict()
        for k, v in checkpoint.items():
            if k in state_dict:
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

    # zeroshot classification as validation split
    val_dataset_label_embeds = {}
    clip_model = None
    for val_dataname, val_dataloader in val_dataloaders.items():
        os.makedirs(config.VAL_DATASET.label_embed_dir, exist_ok=True)
        label_embed_file = os.path.join(config.VAL_DATASET.label_embed_dir, f'{val_dataname}_{config.MODEL.clip_model}.npy')
        if os.path.exists(label_embed_file):
            val_dataset_label_embeds[val_dataname] = torch.from_numpy(np.load(label_embed_file)).to(device)
        else:
            if clip_model is None:
                if config.MODEL.clip_model == 'clip':
                    clip_model = ClipEncoder(device=device)
                elif config.MODEL.clip_model == 'openclip':
                    clip_model = OpenClipEncoder(device=device)
                clip_model.eval()
            with torch.no_grad():
                val_dataset = val_dataloader.dataset
                val_label_embeds = [torch.mean(clip_model('text', label_name), 0) for label_name in val_dataset.category_names]
                val_label_embeds = torch.stack(val_label_embeds, 0)
                val_label_embeds = F.normalize(val_label_embeds, p=2, dim=1)
                val_dataset_label_embeds[val_dataname] = val_label_embeds
            np.save(label_embed_file, val_label_embeds.cpu().numpy())
    if clip_model is not None:
        del clip_model

    for step, (task_name, batch) in enumerate(meta_loader):
        if global_step >= config.TRAIN.num_train_steps:
            break

        # forward pass
        if task_name == 'single_object':
            _, losses = model('mae_csc', batch, compute_loss=True)
        elif task_name == 'grasp':
            _, losses = model('grasp', batch, compute_loss=True)
        elif task_name == 'multi_objects':
            multiple_tasks = []
            if config.MODEL.loss_config.scene_mae_csc_loss_weight > 0:
                multiple_tasks.append('mae_csc')
            if config.MODEL.loss_config.obj_loss_weight > 0 or config.MODEL.loss_config.ref_loss_weight > 0:
                multiple_tasks.append('obj_ref')
            if len(multiple_tasks) == 1:
                multiple_tasks = multiple_tasks[0]
            _, losses = model(multiple_tasks, batch, compute_loss=True)
        else:
            raise NotImplementedError(f'Unknown task name: {task_name}')

        # backward pass
        if config.TRAIN.gradient_accumulation_steps > 1:  # average loss
            losses['total'] = losses['total'] / config.TRAIN.gradient_accumulation_steps
        losses['total'].backward()

        for key, value in losses.items():
            TB_LOGGER.add_scalar(f'step/{task_name}/loss_{key}', value.item(), global_step)
            running_metrics.setdefault(f'{task_name}_loss_{key}', RunningMeter(f'{task_name}_loss_{key}'))
            running_metrics[f'{task_name}_loss_{key}'](value.item())
        
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
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)

        if global_step % config.TRAIN.log_steps == 0:
            # monitor training throughput
            LOGGER.info(f'==============Step {global_step}===============')
            LOGGER.info(', '.join(['%s:%.4f' % (lk, lv.val) for lk, lv in running_metrics.items()]))
            LOGGER.info('===============================================')

        if global_step % config.TRAIN.save_steps == 0:
            model_saver.save(model, global_step, optimizer=optimizer, rewrite_optimizer=True)

        if global_step % config.TRAIN.val_steps == 0:
            for val_dataname, val_dataloader in val_dataloaders.items():           
                val_metrics = validate(model, val_dataloader, val_dataset_label_embeds[val_dataname])
                LOGGER.info(f'=================Validation: {val_dataname}=================')
                LOGGER.info(f'Accuracy: {val_metrics["acc"]:.4f}')
                LOGGER.info('===============================================')
            model.train()

        if global_step >= config.TRAIN.num_train_steps:
            break

    if global_step % config.TRAIN.save_steps != 0:
        LOGGER.info(
            f'==============Step {global_step}===============')
        LOGGER.info(', '.join(['%s:%.4f' % (lk, lv.val) for lk, lv in running_metrics.items()]))
        LOGGER.info('===============================================')
        model_saver.save(model, global_step, optimizer=optimizer, rewrite_optimizer=True)
        for val_dataname, val_dataloader in val_dataloaders.items():           
            val_metrics = validate(model, val_dataloader, val_dataset_label_embeds[val_dataname])
            LOGGER.info(f'=================Validation: {val_dataname}=================')
            LOGGER.info(f'Accuracy: {val_metrics["acc"]:.4f}')
            LOGGER.info('===============================================')


@torch.no_grad()
def validate(model, val_dataloader, val_label_embeds, fusion_ratio=0.25):
    model.eval()
    num_samples = 0
    num_corrects = 0
    for batch in val_dataloader:
        pred_txt_fts, pred_img_fts = model('mae_csc', batch, compute_loss=False, mask_pc=False)
        pred_fts = pred_txt_fts * fusion_ratio + pred_img_fts * (1 - fusion_ratio)
        sims = torch.matmul(
            F.normalize(pred_fts, p=2, dim=1), val_label_embeds.permute(1, 0)
        )
        preds = sims.argmax(dim=1).data.cpu()
        num_corrects += (preds == batch['labels']).float().sum().item()
        num_samples += preds.size(0)
    acc = num_corrects / num_samples
    
    return {'acc': acc}


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
