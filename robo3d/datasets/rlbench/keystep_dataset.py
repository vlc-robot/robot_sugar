from typing import List, Dict, Optional

import os
import numpy as np
import einops
import json

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from robo3d.utils.ops import pad_tensors, gen_seq_masks


class DataTransform(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, data) -> Dict[str, torch.Tensor]:
        """
        Inputs:
            data: dict
                - rgb: (T, N, C, H, W), N: num of cameras
                - pc: (T, N, C, H, W)
        """
        keys = list(data.keys())

        # Continuous range of scales
        sc = np.random.uniform(*self.scales)

        t, n, c, raw_h, raw_w = data[keys[0]].shape
        data = {k: v.flatten(0, 1) for k, v in data.items()} # (t*n, h, w, c)
        resized_size = [int(raw_h * sc), int(raw_w * sc)]

        # Resize based on randomly sampled scale
        data = {
            k: transforms_f.resize(
                v,
                resized_size,
                transforms.InterpolationMode.BILINEAR
            )
            for k, v in data.items()
        }

        # Adding padding if crop size is smaller than the resized size
        if raw_h > resized_size[0] or raw_w > resized_size[1]:
            right_pad = max(raw_w - resized_size[1], 0)
            bottom_pad = max(raw_h - resized_size[0], 0)
            data = {
                k: transforms_f.pad(
                    v,
                    padding=[0, 0, right_pad, bottom_pad],
                    padding_mode="edge",
                )
                for k, v in data.items()
            }

        # Random Cropping
        i, j, h, w = transforms.RandomCrop.get_params(
            data[keys[0]], output_size=(raw_h, raw_w)
        )

        data = {k: transforms_f.crop(v, i, j, h, w) for k, v in data.items()}

        data = {
            k: einops.rearrange(v, "(t n) c h w -> t n c h w", t=t)
            for k, v in data.items()
        }

        return data


class KeystepDataset(Dataset):
    def __init__(
        self, data_dir, taskvars, instr_embed_file=None, 
        gripper_channel=False, camera_ids=None, 
        cameras=('left_shoulder', 'right_shoulder', 'wrist'), 
        use_instr_embed='none', is_training=False, 
        in_memory=False, only_success=False, **kwargs
    ):
        '''
        - use_instr_embed: 
            'none': use task_id; 
            'avg': use the average instruction embedding;
            'last': use the last instruction embedding;
            'all': use the embedding of all instruction tokens.
        '''
        self.data_dir = data_dir

        if isinstance(taskvars, str):
            assert os.path.exists(taskvars), f'path{taskvars} not exists'
            taskvars = json.load(open(taskvars, 'r'))
            if isinstance(taskvars, dict):
                taskvars = list(taskvars.keys())
            assert isinstance(taskvars, list)
            taskvars.sort()
        self.taskvars = taskvars

        self.instr_embed_file = instr_embed_file
        self.taskvar_to_id = {
            x: i for i, x in enumerate(self.taskvars)
        }
        self.use_instr_embed = use_instr_embed
        self.gripper_channel = gripper_channel
        self.cameras = cameras
        if camera_ids is None:
            self.camera_ids = np.arange(len(self.cameras))
        else:
            self.camera_ids = np.array(camera_ids)
        self.in_memory = in_memory
        self.is_training = is_training
        self.multi_instruction = kwargs.get('multi_instruction', True)
        self.max_demos_per_taskvar = kwargs.get('max_demos_per_taskvar', None)
        self.max_episode_steps = kwargs.get('max_episode_steps', None)

        self.memory = {}

        self._transform = DataTransform((0.75, 1.25))
        
        self.lmdb_envs, self.lmdb_txns = [], []
        self.episode_ids = []
        train_success_result_dir = kwargs.get('train_success_result_dir', data_dir)
        if train_success_result_dir is None:
            train_success_result_dir = data_dir
        for i, taskvar in enumerate(self.taskvars):
            demo_res_file = os.path.join(train_success_result_dir, taskvar, 'results.json')
            if only_success and os.path.exists(demo_res_file):
                demo_results = json.load(open(demo_res_file, 'r'))
            if not os.path.exists(os.path.join(data_dir, taskvar)):
                self.lmdb_envs.append(None)
                self.lmdb_txns.append(None)
                continue
            lmdb_env = lmdb.open(os.path.join(data_dir, taskvar), readonly=True, lock=False)
            self.lmdb_envs.append(lmdb_env)
            lmdb_txn = lmdb_env.begin()
            self.lmdb_txns.append(lmdb_txn)
            keys = [key.decode('ascii') for key in list(lmdb_txn.cursor().iternext(values=False))]
            self.episode_ids.extend(
                [(i, key.encode('ascii')) for key in keys \
                 if key.startswith('episode') and \
                 ((not only_success) or demo_results[key])][:self.max_demos_per_taskvar]
            )
            if self.in_memory:
                self.memory[f'taskvar{i}'] = {}

        if self.use_instr_embed != 'none':
            assert self.instr_embed_file is not None
            self.lmdb_instr_env = lmdb.open(self.instr_embed_file, readonly=True, lock=False)
            self.lmdb_instr_txn = self.lmdb_instr_env.begin()
            self.memory['instr_embeds'] = {}
        else:
            self.lmdb_instr_env = None

    def __exit__(self):
        for lmdb_env in self.lmdb_envs:
            if lmdb_env is not None:
                lmdb_env.close()
        if self.lmdb_instr_env is not None:
            self.lmdb_instr_env.close()

    def __len__(self):
        return len(self.episode_ids)
    
    def get_taskvar_episode(self, taskvar_idx, episode_key):
        if self.in_memory:
            mem_key = f'taskvar{taskvar_idx}'
            if episode_key in self.memory[mem_key]:
                return self.memory[mem_key][episode_key]
        
        value = self.lmdb_txns[taskvar_idx].get(episode_key)
        value = msgpack.unpackb(value)
        # rgb, pc: (num_steps, num_cameras, height, width, 3)
        value['rgb'] = value['rgb'][:, self.camera_ids]
        value['pc'] = value['pc'][:, self.camera_ids]
        if self.in_memory:
            self.memory[mem_key][episode_key] = value
        return value

    def get_taskvar_instr_embeds(self, taskvar):
        instr_embeds = None
        if True: #self.in_memory:
            if taskvar in self.memory['instr_embeds']:
                instr_embeds = self.memory['instr_embeds'][taskvar]
        
        if instr_embeds is None:
            instr_embeds = self.lmdb_instr_txn.get(taskvar.encode('ascii'))
            instr_embeds = msgpack.unpackb(instr_embeds)
            instr_embeds = [torch.from_numpy(x).float() for x in instr_embeds]
            if self.in_memory:
                self.memory['instr_embeds'][taskvar] = instr_embeds

        # randomly select one instruction for the taskvar
        if self.multi_instruction:
            ridx = np.random.randint(len(instr_embeds))
        else:
            ridx = 0
        instr_embeds = instr_embeds[ridx]

        if self.use_instr_embed == 'avg':
            instr_embeds = torch.mean(instr_embeds, 0, keepdim=True)
        elif self.use_instr_embed == 'last':
            instr_embeds = instr_embeds[-1:]
        
        return instr_embeds # (num_ttokens, dim)
        
    def __getitem__(self, idx):
        taskvar_idx, episode_key = self.episode_ids[idx]

        value = self.get_taskvar_episode(taskvar_idx, episode_key)

        # The last one is the stop observation
        rgbs = torch.from_numpy(value['rgb'][:-1]).float().permute(0, 1, 4, 2, 3)  # (T, N, C, H, W)
        pcs = torch.from_numpy(value['pc'][:-1]).float().permute(0, 1, 4, 2, 3)
        # normalise to [-1, 1]
        rgbs = 2 * (rgbs / 255.0 - 0.5)

        num_steps, num_cameras, _, im_height, im_width  = rgbs.size()

        if self.gripper_channel:
            gripper_imgs = torch.zeros(
                num_steps, num_cameras, 1, im_height, im_width, 
                dtype=torch.float32
            )
            for t in range(num_steps):
                for c, cam in enumerate(self.cameras):
                    u, v = value['gripper_pose'][t][cam]
                    if u >= 0 and u < 128 and v >=0 and v < 128:
                        gripper_imgs[t, c, 0, v, u] = 1
            rgbs = torch.cat([rgbs, gripper_imgs], dim=2)
        
        # rgb, pcd: (T, N, C, H, W)
        outs = {'rgbs': rgbs, 'pcds': pcs}
        if self.is_training:
            outs = self._transform(outs)

        outs['step_ids'] = torch.arange(0, num_steps).long()        
        outs['actions'] = torch.from_numpy(value['action'][1:])
        outs['episode_ids'] = episode_key.decode('ascii')
        outs['taskvars'] = self.taskvars[taskvar_idx]
        outs['taskvar_ids'] = taskvar_idx

        if self.max_episode_steps is not None:
            for key in ['rgbs', 'pcds', 'step_ids', 'actions']:
                outs[key] = outs[key][:self.max_episode_steps]

        if self.use_instr_embed != 'none':
            outs['instr_embeds'] = self.get_taskvar_instr_embeds(outs['taskvars'])

        return outs


def stepwise_collate_fn(data: List[Dict]):
    batch = {}
    
    for key in data[0].keys():
        if key == 'taskvar_ids':
            batch[key] = [
                torch.LongTensor([v['taskvar_ids']] * len(v['step_ids'])) for v in data
            ]
        elif key == 'instr_embeds':
            batch[key] = sum([
                [v['instr_embeds']] * len(v['step_ids']) for v in data
            ], [])
        else:
            batch[key] = [v[key] for v in data]

    for key in ['rgbs', 'pcds', 'taskvar_ids', 'step_ids', 'actions']:
        # e.g. rgbs: (B*T, N, C, H, W)
        batch[key] = torch.cat(batch[key], dim=0)

    if 'instr_embeds' in batch:
        batch['instr_embeds'] = pad_tensors(batch['instr_embeds'])

    return batch


def episode_collate_fn(data: List[Dict]):
    batch = {}

    for key in data[0].keys():
        batch[key] = [v[key] for v in data]

    batch['taskvar_ids'] = torch.LongTensor(batch['taskvar_ids'])
    num_steps = [len(x['rgbs']) for x in data]
    if 'instr_embeds' in batch:
        num_ttokens = [len(x['instr_embeds']) for x in data]

    for key in ['rgbs', 'pcds', 'step_ids', 'actions']:
        # e.g. rgbs: (B, T, N, C, H, W)
        batch[key] = pad_tensors(batch[key], lens=num_steps)

    if 'instr_embeds' in batch:
        batch['instr_embeds'] = pad_tensors(batch['instr_embeds'], lens=num_ttokens)
        batch['txt_masks'] = torch.from_numpy(gen_seq_masks(num_ttokens))
    else:
        batch['txt_masks'] = torch.ones(len(num_steps), 1).bool()

    batch['step_masks'] = torch.from_numpy(gen_seq_masks(num_steps))
        
    return batch



if __name__ == '__main__':
    import time
    from torch.utils.data import DataLoader

    data_dir = 'data/train_dataset/keysteps/seed0'
    taskvars = ['pick_up_cup+0']
    cameras = ['left_shoulder', 'right_shoulder', 'wrist']
    instr_embed_file = None
    instr_embed_file = 'data/train_dataset/taskvar_instrs/clip'

    dataset = KeystepDataset(data_dir, taskvars, 
        instr_embed_file=instr_embed_file, 
        use_instr_embed='all',
        gripper_channel='attn', cameras=cameras, 
        is_training=True
    )

    data_loader = DataLoader(
        dataset, batch_size=16, 
        # collate_fn=stepwise_collate_fn
        collate_fn=episode_collate_fn,
    )

    print(len(dataset), len(data_loader))

    st = time.time()
    for batch in data_loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.size())
        break
    et = time.time()
    print('cost time: %.2fs' % (et - st))
