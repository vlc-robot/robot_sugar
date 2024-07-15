from typing import List, Dict, Optional

import os
import numpy as np
import cv2
import json
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from robo3d.datasets.pointcloud.common import RandAugmentMC


class ObjectGraspDataset(Dataset):
    def __init__(
        self, dataset_names=None, dataset_cfgs=None,
        num_points=1024, use_color=True, augment_color=False,
        random_z_rotate=True, **kwargs
    ):
        super().__init__()
        
        self.dataset_names = dataset_names
        self.dataset_cfgs = dataset_cfgs
        self.num_points = num_points
        self.use_color = use_color
        self.random_z_rotate = random_z_rotate
        self.augment_color = augment_color

        if augment_color:
            self.color_aug = RandAugmentMC(n=2, m=10)
        else:
            self.color_aug = None
        
        self.pc_lmdb_envs, self.pc_lmdb_txns = {}, {}
        self.grasp_lmdb_envs, self.grasp_lmdb_txns = {}, {}

        self.data_ids = []

        for dataset_name in self.dataset_names:
            cfg = self.dataset_cfgs[dataset_name]

            self.pc_lmdb_envs[dataset_name] = lmdb.open(cfg.pc_file, readonly=True, lock=False)
            self.pc_lmdb_txns[dataset_name] = self.pc_lmdb_envs[dataset_name].begin()

            self.grasp_lmdb_envs[dataset_name] = lmdb.open(cfg.grasp_file, readonly=True, lock=False)
            self.grasp_lmdb_txns[dataset_name] = self.grasp_lmdb_envs[dataset_name].begin()

            data_ids = json.load(open(cfg.data_ids_file, 'r'))
            for data_id in data_ids:
                self.data_ids.append((dataset_name, data_id))

    def __len__(self):
        return len(self.data_ids)

    def __exit__(self):
        super().__exit__()
        for envs in [self.pc_lmdb_envs, self.grasp_lmdb_envs]:
            for env in envs.value():
                env.close()

    def __getitem__(self, idx):
        dataset_name, data_key = self.data_ids[idx]
        dataset_cfg = self.dataset_cfgs[dataset_name]
        enc_data_key = data_key.encode('ascii')

        pc_fts = msgpack.unpackb(self.pc_lmdb_txns[dataset_name].get(enc_data_key))
        grasps = msgpack.unpackb(self.grasp_lmdb_txns[dataset_name].get(enc_data_key))
        grasp_rotations = grasps['rotations']
        grasp_translations = grasps['translations']
        grasp_valid_masks = grasps['valid_points']
        grasp_transform_idxs = grasps['transform_idxs']

        # Sample points
        pc_idxs = np.random.choice(
            pc_fts.shape[0], self.num_points, replace=pc_fts.shape[0] < self.num_points
        )
        pc_fts = pc_fts[pc_idxs]
        grasp_valid_masks = grasp_valid_masks[pc_idxs]
        grasp_transform_idxs = grasp_transform_idxs[pc_idxs]

        xyz = pc_fts[..., :3]
        # centerize pc
        xyz_center = np.mean(xyz, 0)
        xyz = xyz - xyz_center
        # scale pc
        xyz_scale = np.max(np.linalg.norm(xyz, axis=-1))
        xyz = xyz / xyz_scale
        grasp_translations = (grasp_translations - xyz_center) / xyz_scale
        
        if self.random_z_rotate:
            angle = np.random.uniform() * 2 * np.pi
            cosval, sinval = np.cos(angle), np.sin(angle)
            rot = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
            xyz = np.dot(xyz, np.transpose(rot))
            grasp_translations = np.dot(grasp_translations, np.transpose(rot))
            grasp_rotations = np.matmul(rot, grasp_rotations)

        point_grasp_translations = grasp_translations[grasp_transform_idxs]
        point_grasp_rotations = grasp_rotations[grasp_transform_idxs]

        # normalize pc colors
        if self.use_color:
            rgb = pc_fts[..., 3:6]
            if self.color_aug is not None:
                rgb = (rgb * 255).astype(np.uint8)
                rgb = np.asarray(self.color_aug(Image.fromarray(rgb)))
                rgb = rgb / 255.
            rgb = rgb * 2 - 1 # (0, 1) -> (-1, 1)
            pc_fts = np.concatenate([xyz, rgb], axis=-1)
        else:
            pc_fts = xyz

        
        outs = {
            'pc_fts': torch.from_numpy(pc_fts).float(),
            'grasp_offsets': torch.from_numpy(point_grasp_translations - xyz).float(),
            'grasp_rotations': torch.from_numpy(point_grasp_rotations).float(),
            'grasp_valid_masks': torch.from_numpy(grasp_valid_masks),
        }
        
        return outs


def grasp_pretrain_collate_fn(data: List[Dict]):
    batch = {}
    
    for key in data[0].keys():
        batch[key] = [v[key] for v in data]
        batch[key] = torch.stack(batch[key], 0)

    return batch
