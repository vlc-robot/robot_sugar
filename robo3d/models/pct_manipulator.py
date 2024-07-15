from typing import List, Dict, Optional

import einops
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

from .base import BaseModel
from .pc_transformer import MaskedPCTransformer
from .point_ops import three_interpolate_feature
from .transformer import PositionalEncoding


def normalise_quat(x):
    return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)
  
class ActionEmbedding(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()

        self.open_embedding = nn.Embedding(2, hidden_size)
        self.pos_embedding = nn.Linear(3, hidden_size)
        self.rot_embedding = nn.Linear(6, hidden_size)  # sin and cos of the euler angles
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, actions):
        '''
        actions: (batch_size, 8)
        '''
        pos_embeds = self.pos_embedding(actions[..., :3])
        open_embeds = self.open_embedding(actions[..., -1].long())

        rot_euler_angles = R.from_quat(actions[..., 3:7].data.cpu()).as_euler('xyz')
        rot_euler_angles = torch.from_numpy(rot_euler_angles).float().to(actions.device)
        rot_inputs = torch.cat(
            [torch.sin(rot_euler_angles), torch.cos(rot_euler_angles)], -1
        )
        rot_embeds = self.rot_embedding(rot_inputs)

        act_embeds = self.layer_norm(
            pos_embeds + rot_embeds + open_embeds
        )
        return act_embeds
    
class ActionHead(nn.Module):
    def __init__(
        self, action_head_config, hidden_size, dropout=0
    ) -> None:
        super().__init__()
        self.config = action_head_config
        assert self.config.pos_pred_type in ['heatmap', 'regression']
        self.heatmap_norm = action_head_config.get('heatmap_norm', True)

        input_size = hidden_size
        if self.config.cat_pc_fts:
            assert self.config.pos_pred_type == 'heatmap'
            input_size += hidden_size

        self.decoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.02),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 4 + 1 + 3),
        )

        multiscale_fusion_layers = [
            nn.Linear(hidden_size * len(self.config.multiscale_pc_layers), hidden_size),
        ]
        if len(self.config.multiscale_pc_layers) > 1:
            multiscale_fusion_layers.append(nn.LeakyReLU(0.02))
            multiscale_fusion_layers.append(nn.LayerNorm(hidden_size, eps=1e-12))
            multiscale_fusion_layers.append(nn.Linear(hidden_size, hidden_size))
        self.multiscale_fusion = nn.Sequential(*multiscale_fusion_layers)
        
    def forward(self, pc_fts, cur_act_token, pc_poses, pc_centers, pc_radii):
        '''
        - pc_fts: (batch, npoints, dim)
        - cur_act_token: (batch, dim)
        - pcd_poses: (batch, npoints, 3)
        - pc_centers: (batch, 3)
        - pc_radii: (batch, )
        ''' 
        if self.multiscale_fusion is not None:
            pc_fts = self.multiscale_fusion(pc_fts)

        xg_fts = cur_act_token
        if self.config.pos_pred_type == 'heatmap':
            # compute cosine similarity between the point cloud features and the current action token
            xt_logits = torch.bmm(
                F.normalize(pc_fts, p=2, dim=2) if self.heatmap_norm else pc_fts, 
                F.normalize(cur_act_token, p=2, dim=1).unsqueeze(2) if self.heatmap_norm else cur_act_token.unsqueeze(2),
            )
            # predict the translation of the gripper: (batch, npoints, 1)
            if self.config.pc_gumbel_softmax:
                xt_heatmap = F.gumbel_softmax(xt_logits, tau=self.config.heatmap_temp, hard=True, dim=1)
            else:
                xt_heatmap = torch.softmax(xt_logits / self.config.heatmap_temp, dim=1)
            # print(torch.max(xt_heatmap, 1)[0])
            xt = einops.reduce(pc_poses * xt_heatmap, 'b n c -> b c', 'sum')
            if self.config.cat_pc_fts:
                pc_attn_fts = einops.reduce(pc_fts * xt_heatmap, 'b n c -> b c', 'sum')
                xg_fts = torch.cat([xg_fts, pc_attn_fts], dim=1)
            
        xg = self.decoder(xg_fts)
        xr = normalise_quat(xg[..., :4])
        xo = xg[..., 4:5]

        if self.config.pos_pred_type == 'heatmap':
            xt_offset = xg[..., 5:]
            xt = xt + xt_offset
        else:
            xt = xg[..., 5:]
            
        xt = xt * pc_radii + pc_centers

        actions = torch.cat([xt, xr, xo], dim=-1)
        outs = {'actions': actions}

        if self.config.pos_pred_type == 'heatmap':
            outs.update({
                'xt_offset':xt_offset * pc_radii.unsqueeze(2), 
                'xt_heatmap': xt_heatmap.squeeze(-1),
            })

        return outs 

class ActionLoss(object):
    def __init__(self, use_discrete_rot: bool = False, rot_resolution: int = 5):
        self.use_discrete_rot = use_discrete_rot
        if self.use_discrete_rot:
            self.rot_resolution = rot_resolution
            self.rot_classes = 360 // rot_resolution

    def decompose_actions(self, actions, onehot_rot=False):
        pos = actions[..., :3]
        if not self.use_discrete_rot:
            rot = actions[..., 3:7]
            open = actions[..., 7]
        else:
            if onehot_rot:
                rot = actions[..., 3: 6].long()
            else:
                rot = [
                    actions[..., 3: 3 + self.rot_classes],
                    actions[..., 3 + self.rot_classes: 3 + 2*self.rot_classes],
                    actions[..., 3 + 2*self.rot_classes: 3 + 3*self.rot_classes],
                ]
            open = actions[..., -1]
        return pos, rot, open

    def compute_loss(
            self, preds, targets, masks=None,
            heatmap_loss=False, distance_weight=1, heatmap_loss_weight=1,
            pred_heatmap_logits=None, pred_offset=None, pcd_xyzs=None,
            use_heatmap_max=False, use_pos_loss=True
        ) -> Dict[str, torch.Tensor]:
        pred_pos, pred_rot, pred_open = self.decompose_actions(preds)
        tgt_pos, tgt_rot, tgt_open = self.decompose_actions(targets, onehot_rot=True)        

        losses = {}
        losses['pos'] = F.mse_loss(pred_pos, tgt_pos)

        if self.use_discrete_rot:
            losses['rot'] = (F.cross_entropy(pred_rot[0], tgt_rot[:, 0]) + \
                            F.cross_entropy(pred_rot[1], tgt_rot[:, 1]) + \
                            F.cross_entropy(pred_rot[2], tgt_rot[:, 2])) / 3
        else:
            # Automatically matching the closest quaternions (symmetrical solution).
            tgt_rot_ = -tgt_rot.clone()
            rot_loss = F.mse_loss(pred_rot, tgt_rot, reduction='none').mean(-1)
            rot_loss_ = F.mse_loss(pred_rot, tgt_rot_, reduction='none').mean(-1)
            select_mask = (rot_loss < rot_loss_).float()
            losses['rot'] = (select_mask * rot_loss + (1 - select_mask) * rot_loss_).mean()

        losses['open'] = F.binary_cross_entropy_with_logits(pred_open, tgt_open)

        if use_pos_loss:
            losses['total'] = losses['pos'] + losses['rot'] + losses['open']
        else:
            losses['total'] = losses['rot'] + losses['open']

        if heatmap_loss:
            # (batch, npoints, 3)
            tgt_offset = targets[:, :3].unsqueeze(1) - pcd_xyzs 
            dists = torch.norm(tgt_offset, dim=-1)
            if use_heatmap_max:
                tgt_heatmap_index = torch.min(dists, 1)[1]  # (b, )
                
                losses['xt_heatmap'] = F.cross_entropy(
                    pred_heatmap_logits, tgt_heatmap_index
                )
                losses['total'] += losses['xt_heatmap'] * heatmap_loss_weight

                losses['xt_offset'] = F.mse_loss(
                    pred_offset.gather(
                        2, einops.repeat(tgt_heatmap_index, 'b -> b 3').unsqueeze(2)
                    ),
                    tgt_offset.gather(
                        1, einops.repeat(tgt_heatmap_index, 'b -> b 3').unsqueeze(1)
                    )
                )
                losses['total'] += losses['xt_offset']

            else:
                inv_dists = 1 / (1e-12 + dists)**distance_weight

                tgt_heatmap = torch.softmax(inv_dists, dim=1)
                tgt_log_heatmap = torch.log_softmax(inv_dists, dim=1)

                losses['tgt_heatmap_max'] = torch.mean(tgt_heatmap.max(1)[0])

                losses['xt_heatmap'] = F.kl_div(
                    torch.log_softmax(pred_heatmap_logits, dim=-1), tgt_log_heatmap,
                    reduction='batchmean', log_target=True
                )
                losses['total'] += losses['xt_heatmap'] * heatmap_loss_weight

                losses['xt_offset'] = torch.sum(F.mse_loss(
                    pred_offset.permute(0, 2, 1), tgt_offset, 
                    reduction='none'
                ) * tgt_heatmap.unsqueeze(2)) / tgt_offset.size(0) / 3
                # losses['xt_offset'] = F.mse_loss(
                #     pred_offset.permute(0, 2, 1), tgt_offset, reduce='mean'
                # )
                losses['total'] += losses['xt_offset']
            # print({k: v.item() for k, v in losses.items()})

        return losses
  

class PCTManipulator(BaseModel):
    def __init__(
        self, transformer_config, action_head_config, max_steps: int = 20,
        instr_embed_size: int = None, instr_max_tokens: int = 77,
        use_prev_action=False, norm_prev_action_pos=False,
        learnable_step_embedding=True, **kwargs
    ):
        super().__init__()

        self.transformer_config = transformer_config
        self.action_head_config = action_head_config
        self.hidden_size = transformer_config.hidden_size
        self.max_steps = max_steps
        self.learnable_step_embedding = learnable_step_embedding
        self.use_prev_action = use_prev_action
        self.norm_prev_action_pos = norm_prev_action_pos
        self.kwargs = kwargs
        
        self.mae_encoder = MaskedPCTransformer(**transformer_config)

        self.cur_action_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        if self.use_prev_action:
            self.prev_action_embedding = ActionEmbedding(self.hidden_size)
        if self.learnable_step_embedding:
            self.stepid_embedding = nn.Embedding(self.max_steps, self.hidden_size)
        else:
            self.stepid_embedding = PositionalEncoding(self.hidden_size, max_len=self.max_steps)
        # cur_action_token, prev_action_token, stepid_token
        self.query_pos = nn.Parameter(torch.zeros(1, 3, self.hidden_size))

        self.ref_txt_fc = nn.Linear(instr_embed_size, self.hidden_size)
        self.ref_txt_pos = nn.Parameter(torch.zeros(1, instr_max_tokens, self.hidden_size))

        self.head = ActionHead(
            self.action_head_config, self.hidden_size, dropout=0,
        )

        self.loss_fn = ActionLoss(use_discrete_rot=False)

        self.apply(self._init_weights)
        for token in [self.cur_action_embedding, self.query_pos, self.ref_txt_pos]:
            trunc_normal_(token, std=.02)

    def forward(self, batch, compute_loss=False):
        batch = self.prepare_batch(batch)
        device = batch['fts'].device
        batch_size = batch['fts'].size(0)

        if self.norm_prev_action_pos:
            batch['prev_actions'][..., :3] = (batch['prev_actions'][..., :3] - batch['pc_centers']) / batch['pc_radii']

        # encode point cloud
        pc_fts = batch['fts'].transpose(1, 2).contiguous()  # (batch, npoints, dim)
        
        queries, query_poses = [], []
        queries.append(
            self.cur_action_embedding.expand(batch_size, -1, -1) # (B, 1, D)
        )
        if self.use_prev_action:
            queries.append(
                self.prev_action_embedding(batch['prev_actions']).unsqueeze(1) # (B, 1, D)
            )
        queries.append(
            self.stepid_embedding(batch['step_ids']).unsqueeze(1) # (B, 1, D)
        )
        query_poses.append(
            self.query_pos[:, :len(queries)].expand(batch_size, -1, -1) # (B, 2/3, D)
        )

        queries.append(
            self.ref_txt_fc(batch['instr_embeds']) # (B, instr_len, D)
        )
        max_txt_len = batch['instr_embeds'].size(1)
        query_poses.append(
            self.ref_txt_pos[:, :max_txt_len].expand(batch_size, -1, -1) # (B, instr_len, D)
        )
        
        queries = torch.cat(queries, dim=1)  # (B, 3+instr_len, D)
        query_poses = torch.cat(query_poses, dim=1)  # (B, 3+instr_len, D)

        num_noninstr_tokens = 3 if self.use_prev_action else 2
        query_padded_masks = torch.cat(
            [torch.zeros(batch_size, num_noninstr_tokens).bool().to(device), \
             batch['txt_masks'].logical_not()], dim=1
        )
        
        mae_outs = self.mae_encoder(
            pc_fts, query=queries, query_pos=query_poses, 
            query_padded_mask=query_padded_masks, 
            skip_tgt_sa=False, detach_src=False,
            return_multiscale_layers=self.action_head_config.multiscale_pc_layers
        )

        if self.action_head_config.multiscale_pc_layers is None:
            out_pc_fts = mae_outs['pc_vis']
        else:
            out_pc_fts = mae_outs['multiscale_pc_fts']

        if self.action_head_config.pc_upsampling:
            out_pc_poses = pc_fts[..., :3].contiguous()
            out_pc_fts = three_interpolate_feature(
                out_pc_poses, mae_outs['centers'], out_pc_fts
            )
            out_pc_fts = out_pc_fts + self.mae_encoder.point_pos_embedding(out_pc_poses)
        else:
            out_pc_poses = mae_outs['centers']
    
        action_outs = self.head(
            out_pc_fts, mae_outs['query'][:, 0], out_pc_poses,
            batch['pc_centers'], batch['pc_radii']
        )
        actions = action_outs['actions']

        if compute_loss:
            losses = self.loss_fn.compute_loss(
                actions, batch['actions']
            )
            return losses, actions

        return actions

