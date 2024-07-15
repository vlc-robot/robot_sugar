from torch.utils.data import Dataset

import os
import h5py
import torch
import numpy as np
import json

from .common import normalize_pc, random_rotate_z, augment_pc, farthest_point_sample


class ObjaverseLVISDataset(Dataset):
    def __init__(
        self, data_ids_file, data_dir, output_num_points=1024,
        y_up=True, use_color=False, fps_sample=False, 
        normalize=True, random_z_rotate=False, augment=False, **kwargs
    ):
        super().__init__()

        self.data_ids = json.load(open(data_ids_file))
        self.data_dir = data_dir
        
        self.output_num_points = output_num_points
        self.fps_sample = fps_sample
        self.y_up = y_up
        self.use_color = use_color
        self.normalize = normalize
        self.random_z_rotate = random_z_rotate
        self.augment = augment

        self.category_names = sorted(list(set([x['category'] for x in self.data_ids])))
        self.category2idx = {x: i for i, x in enumerate(self.category_names)}
        self.num_categories = len(self.category_names)

    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, idx):
        filepath = self.data_ids[idx]['data_path'].split('/')[-2:]
        filepath = os.path.join(self.data_dir, *filepath)
        data = np.load(filepath, allow_pickle=True).item()

        pc = np.concatenate([data['xyz'], data['rgb']], 1)

        if pc.shape[0] > self.output_num_points:
            if self.fps_sample: # TODO: speedup, this is slow
                pc = farthest_point_sample(pc, self.output_num_points)
            else:
                pc = pc[np.random.choice(pc.shape[0], self.output_num_points, replace=False)]
        elif pc.shape[0] < self.output_num_points:
            pc = pc[np.random.choice(pc.shape[0], self.output_num_points, replace=True)]

        xyz = pc[:, :3]
        if self.y_up:
            # swap y and z axis (gravity direction)
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.augment:
            xyz = augment_pc(xyz)
        if self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        
        if self.use_color:
            rgb = pc[:, 3:6]
            rgb = rgb * 2 - 1   # (0, 1) -> (-1, 1)
            pc = np.concatenate([xyz, rgb], axis=1)
        else:
            pc = xyz

        label = self.category2idx[self.data_ids[idx]['category']]

        return {
            'pc_fts': torch.from_numpy(pc).float(),
            'labels': label
        }

    
def objaverse_lvis_collate_fn(data):
    batch = {}
    for key in data[0].keys():
        batch[key] = [x[key] for x in data]

    batch['pc_fts'] = torch.stack(batch['pc_fts'], 0)
    batch['labels'] = torch.LongTensor(batch['labels'])
    return batch
