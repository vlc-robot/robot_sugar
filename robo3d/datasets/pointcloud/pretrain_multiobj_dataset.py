from typing import List, Dict, Optional

import os
import numpy as np
from PIL import Image
import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from .common import (
    normalize_pc, random_rotate_z, random_rotate_xyz, 
    augment_pc, farthest_point_sample
)
from robo3d.datasets.pointcloud.common import RandAugmentMC
from robo3d.utils.ops import pad_tensors


MIN_OBJ_POINTS = 32

class MultiObjectPretrainDataset(Dataset):
    def __init__(
        self, dataset_names=None, dataset_cfgs=None,
        num_points=4096, use_color=True, normalize=True, 
        random_z_rotate=True, random_xyz_rotate=False,
        augment=True, augment_color=False, fps_sample=False, max_ref_txt_len=77, 
        return_scene_fts=False, return_obj_fts=False, return_ref_fts=False, 
        keep_background_ratio=1.0, **kwargs
    ):
        super().__init__()
        
        self.dataset_names = dataset_names
        self.dataset_cfgs = dataset_cfgs
        self.num_points = num_points
        self.use_color = use_color
        self.normalize = normalize
        self.random_z_rotate = random_z_rotate
        self.random_xyz_rotate = random_xyz_rotate
        self.augment = augment
        self.augment_color = augment_color
        self.fps_sample = fps_sample
        self.keep_background_ratio = keep_background_ratio
        self.max_ref_txt_len = max_ref_txt_len
        self.return_scene_fts = return_scene_fts
        self.return_obj_fts = return_obj_fts
        self.return_ref_fts = return_ref_fts
        self.kwargs = kwargs

        if augment_color:
            self.color_aug = RandAugmentMC(n=2, m=10)
        else:
            self.color_aug = None
        
        self.pc_lmdb_envs, self.obj_img_ft_lmdb_envs, self.obj_txt_ft_lmdb_envs = {}, {}, {}
        self.pc_lmdb_txns, self.obj_img_ft_lmdb_txns, self.obj_txt_ft_lmdb_txns = {}, {}, {}
        self.scene_img_ft_lmdb_envs, self.scene_img_ft_lmdb_txns = {}, {}
        self.scene_txt_ft_lmdb_envs, self.scene_txt_ft_lmdb_txns = {}, {}
        self.ref_txt_ft_lmdb_envs, self.ref_txt_ft_lmdb_txns = {}, {}
        self.data_ids = []

        for dataset_name in self.dataset_names:
            cfg = self.dataset_cfgs[dataset_name]

            self.pc_lmdb_envs[dataset_name] = lmdb.open(cfg.pc_file, readonly=True, lock=False)
            self.pc_lmdb_txns[dataset_name] = self.pc_lmdb_envs[dataset_name].begin()

            if self.return_scene_fts:
                self.scene_img_ft_lmdb_envs[dataset_name] = lmdb.open(cfg.scene_img_ft_file, readonly=True, lock=False)
                self.scene_txt_ft_lmdb_envs[dataset_name] = lmdb.open(cfg.scene_txt_ft_file, readonly=True, lock=False)
                self.scene_img_ft_lmdb_txns[dataset_name] = self.scene_img_ft_lmdb_envs[dataset_name].begin()
                self.scene_txt_ft_lmdb_txns[dataset_name] = self.scene_txt_ft_lmdb_envs[dataset_name].begin()
            
            if self.return_obj_fts:
                self.obj_img_ft_lmdb_envs[dataset_name] = lmdb.open(cfg.obj_img_ft_file, readonly=True, lock=False)
                self.obj_txt_ft_lmdb_envs[dataset_name] = lmdb.open(cfg.obj_txt_ft_file, readonly=True, lock=False)
                self.obj_img_ft_lmdb_txns[dataset_name] = self.obj_img_ft_lmdb_envs[dataset_name].begin()
                self.obj_txt_ft_lmdb_txns[dataset_name] = self.obj_txt_ft_lmdb_envs[dataset_name].begin()
                for k, v in self.obj_img_ft_lmdb_txns[dataset_name].cursor():
                    self.obj_img_ft_dim = msgpack.unpackb(v).shape[-1]
                    break
                for k, v in self.obj_txt_ft_lmdb_txns[dataset_name].cursor():
                    v = msgpack.unpackb(v)
                    if isinstance(v, dict):
                        self.obj_txt_ft_dim = v[list(v.keys())[0]].shape[-1]
                    else:
                        self.obj_txt_ft_dim = v[0].shape[-1]
                    break
            
            if self.return_ref_fts:
                self.ref_txt_ft_lmdb_envs[dataset_name] = lmdb.open(cfg.ref_txt_ft_file, readonly=True, lock=False)
                self.ref_txt_ft_lmdb_txns[dataset_name] = self.ref_txt_ft_lmdb_envs[dataset_name].begin()

            data_ids = json.load(open(cfg.data_ids_file, 'r'))
            # self.pc_lmdb_txns[dataset_name].cursor().iternext(values=False)
            for data_id in data_ids:
                self.data_ids.append((dataset_name, data_id))

    def __len__(self):
        return len(self.data_ids)

    def __exit__(self):
        super().__exit__()
        for envs in [self.pc_lmdb_envs, self.obj_img_ft_lmdb_envs, self.obj_txt_ft_lmdb_envs, self.ref_txt_ft_lmdb_envs]:
            for env in envs.value():
                env.close()

    def sample_txt_fts(self, all_txt_fts, text_source):
        txt_fts = []
        if isinstance(all_txt_fts, dict):
            # Equally sample from different text sources
            for key in text_source:
                source_txt_fts = all_txt_fts[key]
                txt_fts.append(source_txt_fts[np.random.randint(len(source_txt_fts))])
        else:
            txt_fts = all_txt_fts
        txt_ft = txt_fts[np.random.randint(len(txt_fts))]
        return txt_ft
                
    def __getitem__(self, idx):
        start_time = time.time()
        dataset_name, data_key = self.data_ids[idx]
        dataset_cfg = self.dataset_cfgs[dataset_name]
        enc_data_key = data_key.encode('ascii')

        # load_pc_time = time.time()
        pc_data = msgpack.unpackb(
            self.pc_lmdb_txns[dataset_name].get(enc_data_key),
            strict_map_key=False
        )
        pc_fts = pc_data['pc_fts']
        obj_info_dict = pc_data['obj_info']
        # load_pc_time = time.time() - load_pc_time

        # ensure that there are at least MIN_OBJ_POINTS points for each instance
        ref_obj_ids = []
        for k, obj_info in obj_info_dict.items():
            if isinstance(obj_info, dict) and (pc_fts[:, -1] == k).sum() >= MIN_OBJ_POINTS:
                ref_obj_ids.append(k)

        # Sample points
        if self.fps_sample:
            # TODO: it is slow to use fps_sample without gpu
            pc_fts = farthest_point_sample(pc_fts, self.output_num_points)
        else:
            # sample MIN_OBJ_POINTS points for each instance
            min_obj_pc_idxs = []
            for obj_id in ref_obj_ids:
                tmp = np.arange(pc_fts.shape[0])[pc_fts[..., -1] == obj_id]
                min_obj_pc_idxs.append(np.random.choice(tmp, MIN_OBJ_POINTS))
            min_obj_pc_idxs = np.concatenate(min_obj_pc_idxs)
            # sample remaining points
            sampled_pc_idxs_set = set(min_obj_pc_idxs.tolist())
            if np.random.rand() < self.keep_background_ratio:
                pc_idxs = [pc_idx for pc_idx in range(pc_fts.shape[0]) if pc_idx not in sampled_pc_idxs_set]
            else:
                pc_idxs = [pc_idx for pc_idx in range(pc_fts.shape[0]) if pc_idx not in sampled_pc_idxs_set and int(pc_fts[pc_idx, -1]) in ref_obj_ids]
            num_remained_points = self.num_points - len(min_obj_pc_idxs)
            pc_idxs = np.random.choice(
                pc_idxs, num_remained_points, replace=len(pc_idxs) < num_remained_points
            )
            pc_idxs = np.concatenate([min_obj_pc_idxs, pc_idxs], 0)
            pc_fts = pc_fts[pc_idxs]

        # separate pc fts and labels
        inst_labels = pc_fts[..., -1].astype(np.int64)
        pc_fts = pc_fts[:, :-1]

        # normalize pc fts
        xyz = pc_fts[..., :3]
        if dataset_cfg.y_up:
            # swap y and z axis (gravity direction)
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.augment:
            xyz = augment_pc(xyz)
        if self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        if self.random_xyz_rotate:
            xyz = random_rotate_xyz(xyz)
        
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
        }

        # mae and contrastive learning
        if self.return_scene_fts:
            # load_scene_time = time.time()
            txt_fts = msgpack.unpackb(self.scene_txt_ft_lmdb_txns[dataset_name].get(enc_data_key))
            outs['txt_fts'] = torch.from_numpy(self.sample_txt_fts(txt_fts, dataset_cfg.txt_source)).float()
            img_fts = msgpack.unpackb(self.scene_img_ft_lmdb_txns[dataset_name].get(enc_data_key))
            outs['img_fts'] = torch.from_numpy(img_fts[np.random.randint(len(img_fts))]).float()
            # load_scene_time = time.time() - load_scene_time

        # instance segmentation
        if self.return_obj_fts:
            obj_idxs, obj_masks, obj_img_fts, obj_txt_fts = [], [], [], []
            for k, obj_info in obj_info_dict.items():
                mask = inst_labels == k
                if mask.sum() < MIN_OBJ_POINTS:
                    continue
                obj_idxs.append(k)
                obj_masks.append(mask)
                if isinstance(obj_info, dict):
                    obj_ft_key = obj_info['obj_id'].encode('ascii')
                    txt_ft = msgpack.unpackb(self.obj_txt_ft_lmdb_txns[dataset_name].get(obj_ft_key))
                    txt_ft = self.sample_txt_fts(txt_ft, dataset_cfg.txt_source)
                    img_ft = msgpack.unpackb(self.obj_img_ft_lmdb_txns[dataset_name].get(obj_ft_key))
                    img_ft = img_ft[np.random.randint(len(img_ft))]
                    obj_img_fts.append(img_ft)
                    obj_txt_fts.append(txt_ft)
                else:   # background: plane, world
                    obj_img_fts.append(np.zeros((self.obj_img_ft_dim, ), dtype=np.float32))
                    obj_txt_fts.append(np.zeros((self.obj_txt_ft_dim, ), dtype=np.float32))

            obj_masks = np.stack(obj_masks, axis=0)
            obj_img_fts = np.stack(obj_img_fts, axis=0)
            obj_txt_fts = np.stack(obj_txt_fts, axis=0)
            outs.update({
                'obj_img_fts': torch.from_numpy(obj_img_fts).float(),
                'obj_txt_fts': torch.from_numpy(obj_txt_fts).float(),
                'obj_masks': torch.from_numpy(obj_masks).bool(),
            })

        # referring expression
        if self.return_ref_fts:
            # load_ref_time = time.time()
            if len(ref_obj_ids) == 0:
                print(idx, data_key, ref_obj_ids, obj_masks.shape, obj_masks.sum(1))
            ref_obj_id = np.random.choice(ref_obj_ids)
            ref_txt_fts = msgpack.unpackb(self.ref_txt_ft_lmdb_txns[dataset_name].get(
                obj_info_dict[ref_obj_id]['obj_id'].encode('ascii')
            ))
            ridx = np.random.randint(len(ref_txt_fts))
            outs['ref_txt_fts'] = torch.from_numpy(ref_txt_fts[ridx][-self.max_ref_txt_len:])
            outs['ref_txt_lens'] = len(outs['ref_txt_fts'])
            outs['ref_masks'] = torch.from_numpy(inst_labels == ref_obj_id)
            # load_ref_time = time.time() - load_ref_time

        # print(time.time() - start_time, load_pc_time, load_scene_time, load_ref_time)
        return outs


def multiobj_pretrain_collate_fn(data: List[Dict]):
    batch = {}
    
    for key in data[0].keys():
        batch[key] = [v[key] for v in data]
    
    for key in ['pc_fts', 'img_fts', 'txt_fts', 'ref_masks']:
        if key in batch:
            batch[key] = torch.stack(batch[key], 0)
    
    if 'ref_txt_fts' in batch:
        batch['ref_txt_fts'] = pad_tensors(batch['ref_txt_fts'], pad=0)
        batch['ref_txt_lens'] = torch.LongTensor(batch['ref_txt_lens'])
        # print('max', batch['ref_txt_lens'].max())

    return batch
