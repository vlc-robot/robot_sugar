import os
import numpy as np

import torch
from torch.utils.data import Dataset

from .common import normalize_pc, augment_pc, random_rotate_z, farthest_point_sample


class ModelNetDataset(Dataset):
    def __init__(
        self, data_root, num_categories, input_num_points, output_num_points,
        split='test', y_up=True, use_color=False, fps_sample=False, 
        normalize=True, random_z_rotate=False, augment=False, 
        color_pad_value=0, **kwargs
    ):
        self.num_categories = num_categories
        self.input_num_points = input_num_points
        self.output_num_points = output_num_points
        self.y_up = y_up
        self.use_color = use_color
        self.color_pad_value = color_pad_value
        self.fps_sample = fps_sample
        self.normalize = normalize
        self.random_z_rotate = random_z_rotate
        self.augment = augment

        # pc_data: list of array (num_points, 6): xyz, normal
        self.pc_list, self.label_list = np.load(
            os.path.join(data_root, f'modelnet{num_categories}_{split}_{input_num_points}pts_fps.dat'),
            allow_pickle=True
        )
        self.pc_list = [x[:, :3] for x in self.pc_list]
        self.label_list = [x[0] for x in self.label_list]

        # self.category_names = [x.strip().replace('_', ' ') for x in open(os.path.join(data_root, f'modelnet{num_categories}_shape_names.txt'))]
        self.category_names = [x.strip() for x in open(os.path.join(data_root, f'modelnet{num_categories}_shape_names.txt'))]

    def __len__(self):
        return len(self.pc_list)

    def __getitem__(self, idx):
        pc = self.pc_list[idx]
        label = self.label_list[idx]

        if pc.shape[0] > self.output_num_points:
            if self.fps_sample: # TODO: speedup, this is slow
                pc = farthest_point_sample(pc, self.output_num_points)
            else:
                pc = pc[np.random.choice(pc.shape[0], self.output_num_points, replace=False)]
        elif pc.shape[0] < self.output_num_points:
            pc = pc[np.random.choice(pc.shape[0], self.output_num_points, replace=True)]

        if self.y_up:
            # swap y and z axis (gravity direction)
            pc[:, [1, 2]] = pc[:, [2, 1]]
        if self.normalize:
            pc = normalize_pc(pc)
        if self.augment:
            pc = augment_pc(pc)
        if self.random_z_rotate:
            pc = random_rotate_z(pc)
        
        if self.use_color:
            colors = np.zeros((pc.shape[0], 3), dtype=np.float32) + self.color_pad_value
            pc = np.concatenate([pc, colors], axis=1)

        return {
            'pc_fts': torch.from_numpy(pc).float(),
            'labels': label
        }

def modelnet_collate_fn(data):
    batch = {}
    for key in data[0].keys():
        batch[key] = [x[key] for x in data]

    batch['pc_fts'] = torch.stack(batch['pc_fts'], 0)
    batch['labels'] = torch.LongTensor(batch['labels'])
    return batch