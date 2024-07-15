from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import os
import numpy as np
import cv2
import json
from PIL import Image
import time
from tqdm import tqdm
import random

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from robo3d.utils.img_randaug import get_transforms
from robo3d.datasets.pointcloud.common import (
    RandAugmentMC, normalize_pc, random_rotate_z, augment_pc, farthest_point_sample
)


class SingleObjectPretrainDataset(Dataset):
    def __init__(
        self, dataset_names=None, dataset_cfgs=None,
        num_points=1024, use_color=True, augment_color=False,
        normalize=True, random_z_rotate=True, augment=True,
        fps_sample=False, use_raw_image=False, **kwargs
    ):
        super().__init__()
        
        self.dataset_names = dataset_names
        self.dataset_cfgs = dataset_cfgs
        self.num_points = num_points
        self.use_color = use_color
        self.normalize = normalize
        self.random_z_rotate = random_z_rotate
        self.augment = augment
        self.augment_color = augment_color
        self.fps_sample = fps_sample
        self.use_raw_image = use_raw_image

        if augment_color:
            self.color_aug = RandAugmentMC(n=2, m=10)
        else:
            self.color_aug = None
        
        self.pc_lmdb_envs, self.img_ft_lmdb_envs, self.txt_ft_lmdb_envs = {}, {}, {}
        self.pc_lmdb_txns, self.img_ft_lmdb_txns, self.txt_ft_lmdb_txns = {}, {}, {}
        self.data_ids = []

        for dataset_name in self.dataset_names:
            cfg = self.dataset_cfgs[dataset_name]

            self.pc_lmdb_envs[dataset_name] = lmdb.open(cfg.pc_file, readonly=True, lock=False)
            self.img_ft_lmdb_envs[dataset_name] = lmdb.open(cfg.img_ft_file, readonly=True, lock=False)
            self.txt_ft_lmdb_envs[dataset_name] = lmdb.open(cfg.txt_ft_file, readonly=True, lock=False)
            
            self.pc_lmdb_txns[dataset_name] = self.pc_lmdb_envs[dataset_name].begin()
            self.img_ft_lmdb_txns[dataset_name] = self.img_ft_lmdb_envs[dataset_name].begin()
            self.txt_ft_lmdb_txns[dataset_name] = self.txt_ft_lmdb_envs[dataset_name].begin()

            data_ids = json.load(open(cfg.data_ids_file, 'r'))
            # random.shuffle(data_ids)
            # self.pc_lmdb_txns[dataset_name].cursor().iternext(values=False)
            for data_id in data_ids:
                self.data_ids.append((dataset_name, data_id))

    def __len__(self):
        return len(self.data_ids)

    def __exit__(self):
        super().__exit__()
        for envs in [self.pc_lmdb_envs, self.img_ft_lmdb_envs, self.txt_ft_lmdb_envs]:
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
        dataset_name, data_key = self.data_ids[idx]
        dataset_cfg = self.dataset_cfgs[dataset_name]
        enc_data_key = data_key.encode('ascii')

        # st = time.time()
        pc_fts = msgpack.unpackb(self.pc_lmdb_txns[dataset_name].get(enc_data_key))
        txt_fts = msgpack.unpackb(self.txt_ft_lmdb_txns[dataset_name].get(enc_data_key))
        img_fts = msgpack.unpackb(self.img_ft_lmdb_txns[dataset_name].get(enc_data_key))
        # print('load time', time.time() - st)

        # img_fts = img_fts[np.random.randint(len(img_fts))]
        # txt_fts = self.sample_txt_fts(txt_fts, dataset_cfg.txt_source)
        # # print(pc_fts.shape, txt_fts.shape, img_fts.shape)
        # return {'pc_fts': torch.from_numpy(pc_fts), 'txt_fts': torch.from_numpy(txt_fts), 'img_fts': torch.from_numpy(img_fts)}

        # st = time.time()
        # Sample points
        if self.fps_sample:
            # TODO: it is slow to use fps_sample without gpu
            pc_fts = farthest_point_sample(pc_fts, self.output_num_points)
        else:
            pc_idxs = np.random.choice(
                pc_fts.shape[0], self.num_points, replace=pc_fts.shape[0] < self.num_points
            )
            pc_fts = pc_fts[pc_idxs]

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

        # sample img and txt fts
        img_fts = img_fts[np.random.randint(len(img_fts))]
        txt_fts = self.sample_txt_fts(txt_fts, dataset_cfg.txt_source)

        outs = {
            'data_ids': data_key,
            'pc_fts': torch.from_numpy(pc_fts).float(),
            'txt_fts': torch.from_numpy(txt_fts).float(),
        }
        if self.use_raw_image:
            raw_images = cv2.imdecode(img_fts, cv2.IMREAD_COLOR)
            raw_images = get_transforms()['train'](raw_images)
            outs['raw_images'] = raw_images
        else:
            outs['img_fts'] = torch.from_numpy(img_fts).float()
        # print('process time', time.time() - st)
        
        # if torch.isnan(outs['pc_fts']).sum().item() > 0:
        #     print(idx, data_key, 'nan', pc_fts)
        #     return self.__getitem__(np.random.randint(len(self)))

        return outs


def singleobj_pretrain_collate_fn(data: List[Dict]):
    batch = {}
    
    for key in data[0].keys():
        batch[key] = [v[key] for v in data]
        if key != 'data_ids':
            batch[key] = torch.stack(batch[key], 0)

    return batch


if __name__ == '__main__':
    from easydict import EasyDict
    dataset_cfgs = EasyDict(
        objaverse=EasyDict({
        "data_ids_file": 'data3d/pretrain_dataset/objaverse_openshape/train_ids_nolvis.json',
        "pc_file": 'data3d/pretrain_dataset/objaverse_openshape/pc_fts',
        "img_ft_file": 'data3d/pretrain_dataset/objaverse_openshape/openclip_img_fts',
        "txt_ft_file": 'data3d/pretrain_dataset/objaverse_openshape/openclip_txt_fts',
        "txt_source": ['text', 'caption', 'retrieval_text'],
        "y_up": True,
    }))
    dataset = SingleObjectPretrainDataset(
        dataset_names=['objaverse'], dataset_cfgs=dataset_cfgs,
        num_points=4096, use_color=True, augment_color=False,
        normalize=True, random_z_rotate=True, augment=True,
        fps_sample=False, use_raw_image=False,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=2, 
        collate_fn=singleobj_pretrain_collate_fn
    )
    for batch in tqdm(dataloader):
        pass
        # print(batch['pc_fts'].shape, batch['txt_fts'].shape, batch['img_fts'].shape)
        # break