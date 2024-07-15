import os
from torch.utils.data import Dataset

import os
import torch
import numpy as np
import json
import cv2

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from robo3d.utils.ops import pad_tensors
from robo3d.datasets.pointcloud.common import normalize_pc, augment_pc

from pointnet2_ops import pointnet2_utils

from PIL import Image

from robo3d.utils.pc_from_depth import (
    get_intrinsic_matrix, pointcloud_from_depth_and_camera_params
)
from robo3d.datasets.pointcloud.common import RandAugmentMC


class RoborefitDataset(Dataset):
    def __init__(
        self, data_root, split, num_points=10000, normalize=True, 
        augment=False, fps_sample=False, ensure_tgt_points=False, 
        use_color=True, augment_color=False, return_raw_pcd=False, 
        raw_data_root=None, clip_name='clip', **kwargs
    ):
        super().__init__()

        data_ids_file = os.path.join(data_root, split, 'data_ids.json')
        self.data_ids = json.load(open(data_ids_file))

        self.annotations = json.load(open(os.path.join(data_root, split, 'roborefit_%s.json' % split)))

        self.pc_dir = os.path.join(data_root, split, 'scene_pcs')
        txt_dir = os.path.join(data_root, split, f'{clip_name}_txt_fts')
        self.lmdb_txt_env = lmdb.open(txt_dir, readonly=True, lock=False)
        self.lmdb_txt_txn = self.lmdb_txt_env.begin()

        self.split = split
        self.num_points = num_points
        self.normalize = normalize
        self.augment = augment
        self.fps_sample = fps_sample
        self.ensure_tgt_points = ensure_tgt_points
        self.use_color = use_color
        self.return_raw_pcd = return_raw_pcd
        self.raw_data_root = raw_data_root
        self.kwargs = kwargs

        if augment_color:
            self.color_aug = RandAugmentMC(n=2, m=10)
        else:
            self.color_aug = None

        self.intrinsic_matrix = get_intrinsic_matrix(
            640, 480, fov_rad=np.deg2rad(60)
        )

    def __exit__(self):
        self.lmdb_txt_env.close()

    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        anno = self.annotations[data_id]

        scene_name = anno['rgb_path'].split('\\')[-1].split('.')[0]
        mask_name = anno['mask_path'].split('\\')[-1].split('.')[0]

        data = np.load(os.path.join(self.pc_dir, f'{scene_name}.npy'), allow_pickle=True).item()
        pc_fts = data['pc_fts'].astype(np.float32)
        pc_labels = data['pc_labels'][mask_name]

        xyz = pc_fts[:, :3]
        rgb = pc_fts[:, 3:6]

        if xyz.shape[0] >= self.num_points:
            if self.fps_sample:
                fps_idx = pointnet2_utils.furthest_point_sample(
                    torch.from_numpy(xyz).float().unsqueeze(0).cuda(), self.num_points
                ) 
                fps_idx = fps_idx.data.cpu().numpy()[0]
            else:
                fps_idx = np.random.permutation(xyz.shape[0])[:self.num_points]
        else:
            fps_idx = np.random.choice(xyz.shape[0], self.num_points, replace=True)

        if self.ensure_tgt_points:
            if pc_labels[fps_idx].sum() == 0:
                tgt_idxs = np.arange(xyz.shape[0])[pc_labels]
                fps_idx[:len(tgt_idxs)] = tgt_idxs

        xyz = xyz[fps_idx]
        rgb = rgb[fps_idx]
        pc_labels = pc_labels[fps_idx]

        if self.return_raw_pcd:
            raw_rgb = cv2.imread(os.path.join(self.raw_data_root, self.split, 'image', f'{scene_name}.png'))
            rraw_rgbgb = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB) / 255.
            raw_depth = cv2.imread(os.path.join(self.raw_data_root, self.split, 'depth', f'{scene_name}.png'), -1)
            raw_depth = raw_depth / 1000.
            raw_label = cv2.imread(os.path.join(self.raw_data_root, self.split, 'mask', scene_name, f'{mask_name}.png'))
            raw_label = np.sum(raw_label.astype(np.int32), -1) > 0

            raw_xyz = pointcloud_from_depth_and_camera_params(
                raw_depth, np.eye(4), self.intrinsic_matrix
            )

            mask = raw_xyz[..., 2] > 0
            raw_rgb = raw_rgb[mask]
            raw_xyz = raw_xyz[mask]
            raw_label = raw_label[mask]
            # shuffle raw data
            ridxs = np.random.permutation(raw_xyz.shape[0])
            raw_rgb = raw_rgb[ridxs]
            raw_xyz = raw_xyz[ridxs]
            raw_label = raw_label[ridxs]

        if self.normalize:
            xyz, (centroid, radius) = normalize_pc(xyz, return_params=True)
            if self.return_raw_pcd:
                raw_xyz = (raw_xyz - centroid) / radius

        if self.augment:
            xyz = augment_pc(xyz)

        if self.use_color:
            if self.color_aug is not None:
                rgb = (rgb * 255).astype(np.uint8)
                rgb = np.asarray(self.color_aug(Image.fromarray(rgb)))
                rgb = rgb / 255.
            rgb = rgb * 2 - 1
            pc_fts = np.concatenate([xyz, rgb], axis=1)
        else:
            pc_fts = xyz

        txt_embed = msgpack.unpackb(self.lmdb_txt_txn.get(str(data_id).encode('ascii')))
        
        outs = {
            'data_ids': data_id,
            'pc_fts': torch.from_numpy(pc_fts).float(),
            'pc_labels': torch.from_numpy(pc_labels).bool(),
            'txt_embeds': torch.from_numpy(txt_embed).float(),
        }
        if self.return_raw_pcd:
            outs['raw_pc_fts'] = torch.from_numpy(
                np.concatenate([raw_xyz, raw_rgb*2-1], axis=1)
            ).float()
            outs['raw_pc_labels'] = torch.from_numpy(raw_label)
            outs['bboxes_2d'] = anno['bbox']
            if self.normalize:
                outs['centroid'] = torch.from_numpy(centroid)
                outs['radius'] = radius
        return outs
    
def roborefit_collate_fn(data):
    batch = {}
    for key in data[0]:
        batch[key] = [item[key] for item in data]
    for key in ['pc_fts', 'pc_labels']:
        batch[key] = torch.stack(batch[key], dim=0)
    batch['txt_lens'] = [len(item['txt_embeds']) for item in data]
    batch['txt_embeds'] = pad_tensors(batch['txt_embeds'], pad=0)

    if 'raw_pc_fts' in data[0]:
        min_num_points = min([x.size(0) for x in batch['raw_pc_fts']])
        batch['raw_pc_fts'] = torch.stack(
            [x[:min_num_points] for x in batch['raw_pc_fts']], dim=0
        )
        batch['raw_pc_labels'] = torch.stack(
            [x[:min_num_points] for x in batch['raw_pc_labels']], dim=0
        )
    if 'centroid' in data[0]:
        batch['centroid'] = torch.stack(batch['centroid'], dim=0)
        batch['radius'] = torch.FloatTensor(batch['radius'])
    return batch
