import os
import sys
import json
import jsonlines
import argparse
import time
from collections import defaultdict
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

# import numpy after torch: https://github.com/pytorch/pytorch/issues/101188
import numpy as np

from robo3d.utils.logger import LOGGER
from robo3d.utils.misc import set_random_seed

from robo3d.configs.default import get_config

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



DATA_FACTORY = {
    'modelnet10': (ModelNetDataset, modelnet_collate_fn),
    'modelnet40': (ModelNetDataset, modelnet_collate_fn),
    'scanobjectnn': (ScanObjectNNDataset, scanobjectnn_collate_fn),
    'scanobjectnn_nobg': (ScanObjectNNDataset, scanobjectnn_collate_fn),
    'scanobjectnn_hardest': (ScanObjectNNDataset, scanobjectnn_collate_fn),
    'objaverse_lvis': (ObjaverseLVISDataset, objaverse_lvis_collate_fn)
}

def main(config, fusion_ratio=0.25):
    # torch.autograd.set_detect_anomaly(True)

    device = 'cuda'

    seed = config.SEED
    set_random_seed(seed)

    # load data validation set
    val_dataloaders = {}
    for val_dataname in config.VAL_DATASET.dataset_names:
        val_dataset_class, val_dataset_collate_fn = DATA_FACTORY[val_dataname]
        val_dataset = val_dataset_class(**config.VAL_DATASET.dataset_cfgs[val_dataname])
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.TRAIN.val_batch_size, collate_fn=val_dataset_collate_fn, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=False
        )
        LOGGER.info(f'{val_dataname}: #num_data: {len(val_dataset)}, #num_catorgies: {val_dataset.num_categories}')
        val_dataloaders[val_dataname] = val_dataloader

    # Prepare model
    model = PCPretrainModel(config.MODEL).to(device)
    model.eval()
    LOGGER.info("Model: nweights %d nparams %d" % (model.num_parameters))
    
    # Load from checkpoint
    ckpt_dir = os.path.dirname(config.checkpoint)
    all_checkpoint_files = [x for x in os.listdir(ckpt_dir) if x.startswith('model_step')]
    all_checkpoint_files.sort(key=lambda x: -int(x.split('_')[-1].split('.')[0]))
    all_checkpoint_files = all_checkpoint_files[all_checkpoint_files.index(os.path.basename(config.checkpoint)):]
    model_checkpoint_files = []
    for k in range(config.checkpoint_ensemble):
        model_checkpoint_files.append(all_checkpoint_files[k])
    avg_checkpoint = None
    for model_checkpoint_file in model_checkpoint_files:
        checkpoint = torch.load(
            os.path.join(ckpt_dir, model_checkpoint_file), map_location=lambda storage, loc: storage)
        LOGGER.info('Load the %s: (%d params)' % (model_checkpoint_file, len(checkpoint)))
        if avg_checkpoint is None:
            avg_checkpoint = checkpoint
        else:
            for k, v in checkpoint.items():
                avg_checkpoint[k] += v
    for k, v in avg_checkpoint.items():
        if 'num_batches_tracked' not in k:
            avg_checkpoint[k] /= len(model_checkpoint_files)
        else:
            print(k, v)

    new_checkpoint = {}
    state_dict = model.state_dict()
    for k, v in avg_checkpoint.items():
        if k in state_dict:
            new_checkpoint[k] = v
    LOGGER.info('Resumed the model checkpoint (%d params)' % len(new_checkpoint))
    model.load_state_dict(new_checkpoint, strict=config.checkpoint_strict_load)

    # zeroshot classification as validation split
    val_dataset_label_embeds = {}
    clip_model = None
    for val_dataname, val_dataloader in val_dataloaders.items():
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

    output_file = os.path.join(config.output_dir, 'preds', 'zeroshot_cls.jsonl')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    outf = jsonlines.open(output_file, 'a')

    for val_dataname, val_dataloader in val_dataloaders.items():           
        val_metrics = validate(model, val_dataloader, val_dataset_label_embeds[val_dataname], fusion_ratio=fusion_ratio)
        LOGGER.info(f'=================Validation: {val_dataname}=================')
        LOGGER.info(f'Acc top1: {val_metrics["acc"]:.4f}, top5: {val_metrics["acc3"]:.4f}, top5: {val_metrics["acc5"]:.4f}')
        LOGGER.info('===============================================')
        val_metrics.update({'dataset': val_dataname, 'ckpt': config.checkpoint, 'ensemble': config.checkpoint_ensemble})
        outf.write(val_metrics)

    outf.close()


@torch.no_grad()
def validate(model, val_dataloader, val_label_embeds, fusion_ratio=0.25):
    model.eval()
    num_samples = 0
    num_corrects, num_corrects_top3, num_corrects_top5 = 0, 0, 0
    for batch in val_dataloader:
        pred_txt_fts, pred_img_fts = model('mae_csc', batch, compute_loss=False, mask_pc=False)
        pred_fts = pred_txt_fts * fusion_ratio + pred_img_fts * (1 - fusion_ratio)
        sims = torch.matmul(
            F.normalize(pred_fts, p=2, dim=1), val_label_embeds.permute(1, 0)
        )
        preds = sims.argsort(dim=1, descending=True).data
        num_corrects += (preds[:, 0] == batch['labels']).float().sum().item()
        num_corrects_top3 += (preds[:, :3] == batch['labels'].unsqueeze(1)).any(dim=1).float().sum().item()
        num_corrects_top5 += (preds[:, :5] == batch['labels'].unsqueeze(1)).any(dim=1).float().sum().item()
        num_samples += preds.size(0)
    acc = num_corrects / num_samples
    
    return {'acc': acc, 'acc3': num_corrects_top3 / num_samples, 'acc5': num_corrects_top5 / num_samples}


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument('--fusion_ratio', type=float, default=0.25)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line (use , to separate values in a list)",
    )
    args = parser.parse_args()

    config = get_config(args.exp_config, args.opts)

    return config, args.fusion_ratio


if __name__ == '__main__':
    config, fusion_ratio = build_args()
    main(config, fusion_ratio)
