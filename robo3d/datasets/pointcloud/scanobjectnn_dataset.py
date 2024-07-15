from torch.utils.data import Dataset

import os
import h5py
import torch
import numpy as np

from .common import normalize_pc, augment_pc, random_rotate_z, farthest_point_sample


class ScanObjectNNDataset(Dataset):
    def __init__(
        self, data_root, split='test', no_bg=False, hardest_setup=False, output_num_points=1024,
        y_up=True, use_color=False, color_pad_value=0.0, fps_sample=False, 
        normalize=True, random_z_rotate=False, augment=False, **kwargs
    ):
        super().__init__()

        self.data_root = data_root
        self.split = split
        self.no_bg = no_bg
        self.hardest_setup = hardest_setup
        
        self.output_num_points = output_num_points
        self.fps_sample = fps_sample
        self.y_up = y_up
        self.use_color = use_color
        self.color_pad_value = color_pad_value
        self.normalize = normalize
        self.random_z_rotate = random_z_rotate
        self.augment = augment

        assert self.split in ['train', 'test'], 'split must be train or test'
        
        if self.hardest_setup:
            if self.split == 'train':
                file_name = 'main_split/training_objectdataset_augmentedrot_scale75.h5'
            else:
                file_name = 'main_split/test_objectdataset_augmentedrot_scale75.h5'
        else:
            if self.no_bg:
                if self.split == 'train':
                    file_name = 'main_split_nobg/training_objectdataset.h5'
                else:
                    file_name = 'main_split_nobg/test_objectdataset.h5'
            else:
                if self.split == 'train':
                    file_name = 'main_split/training_objectdataset.h5'
                else:
                    file_name = 'main_split/test_objectdataset.h5'

        with h5py.File(os.path.join(self.data_root, file_name), 'r') as f:
            self.pc_list = np.array(f['data']).astype(np.float32)   # (num_data, num_points, 3)
            self.label_list = np.array(f['label']).astype(int)      # (num_data,)

        self.category_names = [
            'bag', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display', 'door', 
            'shelf', 'table', 'bed', 'pillow', 'sink', 'sofa', 'toilet'
        ]
        self.num_categories = len(self.category_names)

    def __len__(self):
        return len(self.pc_list)
    
    def __getitem__(self, idx):
        pc = self.pc_list[idx]
        if pc.shape[-1] > 3:
            xyz = pc[:, :3]
            rgb = pc[:, 3:6]
        else:
            xyz = pc
            rgb = None

        label = self.label_list[idx]

        if xyz.shape[0] > self.output_num_points:
            if self.fps_sample: # TODO: speedup, this is slow
                xyz = farthest_point_sample(xyz, self.output_num_points)
            else:
                fps_idxs = np.random.choice(xyz.shape[0], self.output_num_points, replace=False)
        elif xyz.shape[0] < self.output_num_points:
            fps_idxs = np.random.choice(xyz.shape[0], self.output_num_points, replace=True)
        else:
            fps_idxs = np.arange(xyz.shape[0])

        xyz = xyz[fps_idxs]
        if rgb is not None:
            rgb = rgb[fps_idxs]

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
            if rgb is None:
            # if True:
                rgb = np.zeros((xyz.shape[0], 3), dtype=np.float32) + self.color_pad_value
            else:
                rgb = rgb * 2 - 1   # (0, 1) -> (-1, 1)
            pc = np.concatenate([xyz, rgb], axis=1)
        else:
            pc = xyz

        return {
            'pc_fts': torch.from_numpy(pc).float(),
            'labels': label
        }

    
def scanobjectnn_collate_fn(data):
    batch = {}
    for key in data[0].keys():
        batch[key] = [x[key] for x in data]

    batch['pc_fts'] = torch.stack(batch['pc_fts'], 0)
    batch['labels'] = torch.LongTensor(batch['labels'])
    return batch
