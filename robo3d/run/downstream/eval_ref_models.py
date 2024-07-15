import os
import sys
import json
import argparse
import time
from collections import defaultdict
from tqdm import tqdm
import copy
import jsonlines

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from robo3d.utils.misc import set_random_seed

from robo3d.configs.default import get_config

from robo3d.datasets.loader import build_dataloader
from robo3d.datasets.pointcloud.roborefit_dataset import (
    RoborefitDataset, roborefit_collate_fn
)
from robo3d.datasets.pointcloud.ocidref_dataset import (
    OCIDRefDataset, ocidref_collate_fn
)

from robo3d.models.pct_ref_models import PCTRefModel
from robo3d.models.point_ops import three_interpolate_feature


dataset_factory = {
    'roborefit': (RoborefitDataset, roborefit_collate_fn),
    'ocidref': (OCIDRefDataset, ocidref_collate_fn),
}

def main(config):
    device = 'cuda'
    set_random_seed(config.SEED)

    config.defrost()
    config.MODEL.freeze_all_except_head = getattr(config.MODEL, 'freeze_all_except_head', False)
    config.freeze()
    print(config)

    # load data training set
    dataset_class, dataset_collate_fn = dataset_factory[config.DATASET.dataset_class]

    val_dataset_config = copy.deepcopy(config.DATASET)
    val_dataset_config.defrost()
    val_dataset_config.augment_color = False
    val_dataset_config.return_raw_pcd = True
    if config.DATASET.dataset_class == 'roborefit':
        val_dataset_config.raw_data_root = 'data3d/VLGrasp/final_dataset'
        val_dataset_config.freeze()
        val_datasets = {
            split_name: dataset_class(**val_dataset_config, split=split_name, augment=False) \
                for split_name in ['testA', 'testB']
        }
        print(f"#num_testA: {len(val_datasets['testA'])}, #num_testB: {len(val_datasets['testB'])}")
    elif config.DATASET.dataset_class == 'ocidref':
        val_dataset_config.raw_data_root = 'data3d/OCID/OCID-dataset'
        val_dataset_config.freeze()
        val_datasets = {
            split_name: dataset_class(**val_dataset_config, split=split_name, augment=False) \
                for split_name in ['val', 'test']
        }
        print(f"#num_val: {len(val_datasets['val'])}, #num_test: {len(val_datasets['test'])}")

    val_dataloaders = {
        val_dataname: torch.utils.data.DataLoader(
            val_dataset, batch_size=config.TRAIN.val_batch_size, shuffle=False,
            num_workers=config.TRAIN.n_workers, pin_memory=True, collate_fn=dataset_collate_fn
        ) for val_dataname, val_dataset in val_datasets.items()
    }

    # Prepare model
    model = PCTRefModel(config.MODEL).to(device)
    model.eval()
    print("Model: nweights %d nparams %d" % (model.num_parameters))
    
    # Load from checkpoint
    model_checkpoint_file = config.checkpoint
    if model_checkpoint_file is not None:
        checkpoint = torch.load(
            model_checkpoint_file, map_location=lambda storage, loc: storage)
        print('Load the model checkpoint (%d params)' % len(checkpoint))
        model_checkpoint = model.state_dict()
        for k, v in model_checkpoint.items():
            if k in checkpoint and v.size() == checkpoint[k].size():
                model_checkpoint[k] = checkpoint[k]
            else:
                print('Skip loading parameter', k)
        model.load_state_dict(model_checkpoint, strict=True)
    
    pred_dir = os.path.join(config.output_dir, 'preds', os.path.basename(model_checkpoint_file).split('.')[0])
    os.makedirs(pred_dir, exist_ok=True)
    for val_dataname, val_dataloader in val_dataloaders.items():
        val_output_file = os.path.join(pred_dir, f'{val_dataname}.json')
        val_metrics = validate(model, val_dataloader, device=device, output_file=val_output_file)
        print(f'=================Validation: {val_dataname}=================')
        metric_str = ', '.join(['%s: %.4f' % (lk, lv) for lk, lv in val_metrics.items()])
        print(metric_str)
        print('===============================================')
        with jsonlines.open(os.path.join(pred_dir, 'scores.jsonl'), 'a') as outf:
            val_metrics.update({'dataname': val_dataname})
            outf.write(val_metrics)

        
@torch.no_grad()
def validate(model, val_dataloader, device='cuda', output_file=None):
    intrinsic_matrix = torch.from_numpy(val_dataloader.dataset.intrinsic_matrix).float().to(device)

    model.eval()
    data_ids, ious, raw_ious, ious_2d = [], [], [], []
    for batch in tqdm(val_dataloader):
        data_ids.extend(batch['data_ids'])
        pred_masks = model(batch, compute_loss=False)
        sigmoid_pred_masks = torch.sigmoid(pred_masks) > 0.5
        pc_labels = batch['pc_labels'].float()
        numerator = (sigmoid_pred_masks * pc_labels).sum(dim=1)
        denominator = (sigmoid_pred_masks.sum(dim=1) + pc_labels.sum(dim=1) - numerator).clip(min=1)
        ious.append((numerator / denominator).cpu().numpy())

        raw_pred_masks = three_interpolate_feature(
            batch['raw_pc_fts'][..., :3].contiguous(), 
            batch['pc_fts'][..., :3].contiguous(), 
            pred_masks.float().unsqueeze(2),
        )[..., 0]
        sigmoid_raw_pred_masks = torch.sigmoid(raw_pred_masks) > 0.5
        raw_pc_labels = batch['raw_pc_labels'].float()
        numerator = (sigmoid_raw_pred_masks * raw_pc_labels).sum(dim=1)
        denominator = (sigmoid_raw_pred_masks.sum(dim=1) + raw_pc_labels.sum(dim=1) - numerator).clip(min=1)
        raw_ious.append((numerator / denominator).cpu().numpy())
        
        for b in range(pred_masks.size(0)):
            if sigmoid_raw_pred_masks[b].sum() > 0:
                obj_xyz = batch['raw_pc_fts'][b][sigmoid_raw_pred_masks[b]][..., :3]
                obj_xyz = obj_xyz * batch['radius'][b] + batch['centroid'][b]
                obj_pixel_coords = torch.matmul(obj_xyz, intrinsic_matrix.transpose(1, 0)) / obj_xyz[..., 2:]
                obj_min_bound = obj_pixel_coords.min(dim=0)[0][:2]
                obj_max_bound = obj_pixel_coords.max(dim=0)[0][:2]
                bbox_2d = torch.cat([obj_min_bound, obj_max_bound], dim=0).cpu().numpy()
                iou_2d = compute_iou_2d(bbox_2d, batch['bboxes_2d'][b])
            else:
                iou_2d = 0
            ious_2d.append(iou_2d)
        # break

    ious = np.concatenate(ious)
    raw_ious = np.concatenate(raw_ious)
    ious_2d = np.array(ious_2d)

    if output_file is not None:
        with open(output_file, 'w') as outf:
            json.dump({
                'data_ids': data_ids, 'ious': ious.tolist(),
                'raw_ious': raw_ious.tolist(), 
                'ious_2d': ious_2d.tolist(),
            }, outf)
    
    metrics = {
        'mIoU': np.mean(ious).item(), 
        'acc_25': np.mean(ious > 0.25).item(), 
        'acc_50': np.mean(ious > 0.5).item(),
        'raw_mIoU': np.mean(raw_ious).item(), 
        'raw_acc_25': np.mean(raw_ious > 0.25).item(), 
        'raw_acc_50': np.mean(raw_ious > 0.5).item(),
        'mIoU_2d': np.mean(ious_2d).item(), 
        'acc_25_2d': np.mean(ious_2d > 0.25).item(), 
        'acc_50_2d': np.mean(ious_2d > 0.5).item(),
    }

    return metrics

def compute_iou_2d(bbox_2d_1, bbox_2d_2):
    x1, y1, x2, y2 = bbox_2d_1
    a1, b1, a2, b2 = bbox_2d_2
    numerator = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
    denominator = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - numerator
    return numerator / denominator

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

    return config


if __name__ == '__main__':
    config = build_args()
    main(config)
