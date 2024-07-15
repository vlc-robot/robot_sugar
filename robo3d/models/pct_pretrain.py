from typing import List, Dict, Optional

import einops
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from timm.models.layers import trunc_normal_

from .base import BaseModel
from .pc_transformer import MaskedPCTransformer
from .point_ops import three_interpolate_feature
from .transformer import PositionalEncoding, SelfAttentionBlock

from chamferdist import ChamferDistance

from scipy.optimize import linear_sum_assignment


class ContrastiveHead(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, query_embeds, key_embeds):
        similarity = torch.matmul(
            F.normalize(query_embeds, dim=-1, p=2),
            F.normalize(key_embeds, dim=-1, p=2).transpose(0, 1)
        ) / self.temperature
        batch_size = similarity.size(0)
        labels = torch.arange(batch_size).to(similarity.device)
        loss = self.criterion(similarity, labels)
        return loss
    
class MAEDecoder(nn.Module):
    def __init__(self, hidden_size, num_heads, depth, dpr):
        super().__init__()
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                hidden_size, num_heads, drop_path=dpr[k], qkv_bias=False
            ) for k in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, x, pos_embeds):
        for block in self.blocks:
            x = block(x + pos_embeds)
        x = self.norm(x)
        return x

class PCPretrainModel(BaseModel):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = config.transformer_config.hidden_size

        self.mae_encoder = MaskedPCTransformer(**config.transformer_config)

        # MAE decoder
        self.masked_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.hidden_size)
        )
        mae_decoder_cfg = self.config.mae_decoder_config
        dpr = [x.item() for x in torch.linspace(0, mae_decoder_cfg.drop_path_rate, mae_decoder_cfg.depth)]
        self.mae_decoder = MAEDecoder(
            self.hidden_size, mae_decoder_cfg.num_heads, mae_decoder_cfg.depth, dpr
        )
        self.mae_decoder_head = nn.Linear(
            self.hidden_size, 
            self.config.transformer_config.group_size * self.config.transformer_config.input_size
        )

        # Cross-modal contrastive
        self.img_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.img_pos = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.txt_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.txt_pos = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        self.img_proj_head = nn.Linear(
            self.hidden_size, self.config.cross_modal_config.img_ft_size
        )
        self.txt_proj_head = nn.Linear(
            self.hidden_size, self.config.cross_modal_config.txt_ft_size
        )

        # instance segmentation
        obj_cfg = config.obj_decoder_config
        self.obj_tokens = nn.Parameter(torch.zeros(1, obj_cfg.num_objects, self.hidden_size))
        self.obj_poses = nn.Parameter(torch.zeros(1, obj_cfg.num_objects, self.hidden_size))
        # predict object mask, confidence score, image feature, text feature
        self.obj_proj_head = nn.Linear(
            self.hidden_size, 
            self.hidden_size + 1 + self.config.cross_modal_config.img_ft_size + self.config.cross_modal_config.txt_ft_size
        )

        # referring expression
        ref_cfg = config.ref_decoder_config
        self.ref_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.ref_pos = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.ref_txt_fc = nn.Linear(ref_cfg.txt_ft_size, self.hidden_size)
        self.ref_txt_pos = nn.Parameter(torch.zeros(1, ref_cfg.max_txt_len, self.hidden_size))
        self.ref_proj_head = nn.Linear(self.hidden_size, self.hidden_size)   

        # grasp pose prediction
        self.grasp_proj_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1 + 3 + 6)  # success, translation, rotation
        )

        self.apply(self._init_weights)
        for token in [self.masked_token, self.img_token, self.img_pos, self.txt_token, self.txt_pos]:
            trunc_normal_(token, std=.02)
        for token in [self.obj_tokens, self.obj_poses, self.ref_token, self.ref_pos, self.ref_txt_pos]:
            trunc_normal_(token, std=.02)

        self.build_loss_func(self.config.loss_config)

    def build_loss_func(self, loss_cfg):
        self.mae_loss_func = ChamferDistance()
        
        # cross-modal loss
        if loss_cfg.csc_loss_type == 'contrastive':
            self.csc_loss_func = ContrastiveHead(temperature=loss_cfg.csc_temperature)
        else:
            self.csc_loss_func = nn.SmoothL1Loss()
    
    def forward(self, task_name, batch, compute_loss=False, mask_pc=True):
        if task_name == 'mae_csc':
            return self.forward_mae_csc(
                batch, compute_loss=compute_loss, mask_pc=mask_pc
            )
        elif task_name == 'obj_ref':
            return self.forward_obj_ref(
                batch, compute_loss=compute_loss, 
            )
        elif task_name == 'grasp':
            return self.forward_grasp(
                batch, compute_loss=compute_loss, 
            )
        else: # joint training
            _, losses_1 = self.forward_mae_csc(
                batch, compute_loss=compute_loss, mask_pc=mask_pc
            )
            _, losses_2 = self.forward_obj_ref(
                batch, compute_loss=compute_loss, 
            )
            for k, v in losses_2.items():
                if k in losses_1:
                    losses_1[k] += v
                losses_1[k] = v
            return None, losses_1
        
    def forward_mae_csc(self, batch, compute_loss=False, mask_pc=True):
        '''batch data:
            pc_fts: (batch, npoints, dim)
            img_fts: (batch, img_dim)
            txt_fts: (batch, txt_dim)
        '''
        batch = self.prepare_batch(batch)
        batch_size = batch['pc_fts'].size(0)
        
        # encode point cloud
        mae_enc_outs = self.mae_encoder(
            batch['pc_fts'], mask_pc=mask_pc, return_ca_inputs=True,
        )

        losses = {'total': 0}

        # mae decoder
        pc_vis = mae_enc_outs['pc_vis']
        centers = mae_enc_outs['centers']   # (batch, num_groups, 3)
        neighborhoods = mae_enc_outs['neighborhoods']   # (batch, num_groups, group_size, dim)
        pc_bool_masks = mae_enc_outs['pc_bool_masks']   # (batch, num_groups)

        pc_vis_pos_embeds = self.decoder_pos_embed(centers[~pc_bool_masks]).reshape(batch_size, -1, self.hidden_size)
        pc_masked_pos_embeds = self.decoder_pos_embed(centers[pc_bool_masks]).reshape(batch_size, -1, self.hidden_size)
        num_masked_tokens = pc_masked_pos_embeds.size(1)
        masked_tokens = self.masked_token.expand(batch_size, num_masked_tokens, -1)

        dec_inputs = torch.cat([pc_vis, masked_tokens], dim=1)
        dec_pos_embeds = torch.cat([pc_vis_pos_embeds, pc_masked_pos_embeds], dim=1)

        # (batch, seq_len, dim)
        dec_outs = self.mae_decoder(dec_inputs, dec_pos_embeds)

        masked_outs = dec_outs[:, -num_masked_tokens:] # (batch_size, num_masked_tokens, dim)
        rebuilt_points = self.mae_decoder_head(masked_outs).reshape(
            -1, self.config.transformer_config.group_size, self.config.transformer_config.input_size
        )

        gt_points = neighborhoods[pc_bool_masks]

        if compute_loss and (self.config.loss_config.mae_loss_weight > 0 or self.config.loss_config.mae_color_loss_weight > 0):
            losses['mdm'], losses['mdm_colors'] = self.mae_loss_func(
                rebuilt_points, gt_points, bidirectional=True, batch_reduction='mean',
                point_reduction='mean', loc_dim_size=3
            )
            losses['total'] += losses['mdm'] * self.config.loss_config.mae_loss_weight
            if self.config.loss_config.mae_color_loss_weight > 0:
                losses['total'] += losses['mdm_colors'] * self.config.loss_config.mae_color_loss_weight
            else:
                del losses['mdm_colors']

        # cross-modal contrastive: text
        pc_layer_outs = mae_enc_outs['ca_inputs']
        if self.config.loss_config.csc_txt_loss_weight > 0:
            txt_tokens = self.txt_token.expand(batch_size, 1, -1)
            txt_pos_embeds = self.txt_pos.expand(batch_size, 1, -1)
            pc_txt_outs = self.mae_encoder.update_query_given_pc_layer_outs(
                pc_layer_outs, txt_tokens, txt_pos_embeds,
                query_padded_mask=None, 
                skip_tgt_sa=self.config.transformer_config.csc_skip_dec_sa, 
                detach_src=self.config.transformer_config.detach_enc_dec
            )
            pc_txt_outs = self.txt_proj_head(pc_txt_outs).squeeze(dim=1)
            if compute_loss:
                losses['csc_txt'] = self.csc_loss_func(pc_txt_outs, batch['txt_fts'])
                losses['total'] += losses['csc_txt'] * self.config.loss_config.csc_txt_loss_weight 
        else:
            pc_txt_outs = None
        
        # cross-modal contrastive: image
        if self.config.loss_config.csc_img_loss_weight > 0:
            img_tokens = self.img_token.expand(batch_size, 1, -1)
            img_pos_embeds = self.img_pos.expand(batch_size, 1, -1)
            pc_img_outs = self.mae_encoder.update_query_given_pc_layer_outs(
                pc_layer_outs, img_tokens, img_pos_embeds,
                query_padded_mask=None, 
                skip_tgt_sa=self.config.transformer_config.csc_skip_dec_sa, 
                detach_src=self.config.transformer_config.detach_enc_dec
            )
            pc_img_outs = self.img_proj_head(pc_img_outs).squeeze(dim=1)
            if compute_loss:
                losses['csc_img'] = self.csc_loss_func(pc_img_outs, batch['img_fts'])
                losses['total'] += losses['csc_img'] * self.config.loss_config.csc_img_loss_weight
        else:
            pc_img_outs = None
        
        # print(losses)
        if compute_loss:
            return (pc_txt_outs, pc_img_outs), losses
        else:
            return (pc_txt_outs, pc_img_outs)

    def forward_obj_ref(self, batch, compute_loss=False):
        '''batch data:
            pc_fts: (batch, npoints, dim)
            obj_img_fts: list of (num_objs, img_dim)
            obj_txt_fts: list of (num_objs, txt_dim)
            obj_masks: list of (num_objs, npoints)
            ref_txt_fts: (batch, max_txt_len, txt_dim)
            ref_txt_lens: (batch, )
            ref_masks: (batch, npoints)
        '''
        batch = self.prepare_batch(batch)
        device = batch['pc_fts'].device
        batch_size = batch['pc_fts'].size(0)

        for key, value in batch.items():
            if key in ['obj_txt_fts', 'obj_img_fts', 'obj_masks']:
                batch[key] = [x.to(device) for x in value]

        # encode point cloud
        mae_enc_outs = self.mae_encoder(
            batch['pc_fts'], mask_pc=False, return_ca_inputs=True,
        )
        pc_vis = mae_enc_outs['pc_vis']
        centers = mae_enc_outs['centers']   # (batch, num_groups, 3)

        enc_pc_poses = batch['pc_fts'][..., :3].contiguous()
        enc_pc_fts = three_interpolate_feature(
            enc_pc_poses, centers, pc_vis
        )
        # enc_pc_fts = enc_pc_fts.detach()

        pc_layer_outs = mae_enc_outs['ca_inputs']

        losses = {}
        task_outs = {}

        # object detection decoder
        if self.config.loss_config.obj_loss_weight > 0:
            obj_tokens = self.obj_tokens.expand(batch_size, -1, -1)
            obj_pos_embeds = self.obj_poses.expand(batch_size, -1, -1)
            obj_outs = self.mae_encoder.update_query_given_pc_layer_outs(
                pc_layer_outs, obj_tokens, obj_pos_embeds,
                query_padded_mask=None, skip_tgt_sa=False, 
                detach_src=self.config.transformer_config.detach_enc_dec
            )
            obj_outs = self.obj_proj_head(obj_outs)

            obj_masks = torch.einsum(
                'bod,bpd->bop', obj_outs[..., :self.hidden_size], enc_pc_fts
            )
            
            obj_conf_scores = obj_outs[..., self.hidden_size]
            obj_img_fts = obj_outs[..., self.hidden_size+1:self.hidden_size+1+self.config.cross_modal_config.img_ft_size]
            obj_txt_fts = obj_outs[..., -self.config.cross_modal_config.txt_ft_size:]
            task_outs['obj'] = (obj_masks, obj_conf_scores, obj_img_fts, obj_txt_fts)
            
            if compute_loss:
                # computing this loss is time consuming
                losses = self.compute_instance_segmentation_loss(
                    obj_masks, obj_conf_scores, obj_img_fts, obj_txt_fts,
                    batch['obj_masks'], batch['obj_img_fts'], batch['obj_txt_fts']
                )
                losses['total'] *= self.config.loss_config.obj_loss_weight

        # refering expression decoder
        if self.config.loss_config.ref_loss_weight > 0:
            ref_token = self.ref_token.expand(batch_size, -1, -1)
            ref_pos_embed = self.ref_pos.expand(batch_size, -1, -1)
            ref_txt_tokens = self.ref_txt_fc(batch['ref_txt_fts'])
            max_txt_len = ref_txt_tokens.size(1)
            ref_txt_pos_embed = self.ref_txt_pos[:, :max_txt_len].expand(batch_size, -1, -1)

            query_tokens = torch.cat([ref_token, ref_txt_tokens], dim=1)
            query_pos_embeds = torch.cat([ref_pos_embed, ref_txt_pos_embed], dim=1)
            query_padded_mask = torch.zeros(batch_size, query_tokens.size(1), dtype=torch.bool)
            for i, txt_len in enumerate(batch['ref_txt_lens']):
                query_padded_mask[i, txt_len+1:] = True
            query_padded_mask = query_padded_mask.to(device)

            ref_outs = self.mae_encoder.update_query_given_pc_layer_outs(
                pc_layer_outs, query_tokens, query_pos_embeds,
                query_padded_mask=query_padded_mask, skip_tgt_sa=False, 
                detach_src=self.config.transformer_config.detach_enc_dec
            )
            ref_outs = self.ref_proj_head(ref_outs[:, 0])
            ref_pred_masks = torch.einsum(
                'bd,bpd->bp', ref_outs, enc_pc_fts
            )
            task_outs['ref'] = ref_pred_masks
            
            if compute_loss:
                ref_loss_cfg = self.config.loss_config.ref_loss
                pc_labels = batch['ref_masks'].float()
                if ref_loss_cfg.focal_loss:
                    losses['ref_bce'] = torchvision.ops.sigmoid_focal_loss(
                        ref_pred_masks, pc_labels, alpha=0.25, gamma=2, reduction='mean'
                    )
                else:
                    losses['ref_bce'] = F.binary_cross_entropy_with_logits(
                        ref_pred_masks, pc_labels, reduction='mean'
                    )
                ref_pred_masks_sigmoid = torch.sigmoid(ref_pred_masks)
                numerator = 2 * (ref_pred_masks_sigmoid * pc_labels).sum(dim=-1)
                denominator = ref_pred_masks_sigmoid.sum(dim=-1) + pc_labels.sum(dim=-1)
                mask_dice = 1 - (numerator + 1) / (denominator + 1)
                losses['ref_dice'] = mask_dice.mean()
                ref_loss = ref_loss_cfg.bce_loss_weight * losses['ref_bce'] + \
                    ref_loss_cfg.dice_loss_weight * losses['ref_dice']
                if 'total' in losses:
                    losses['total'] += ref_loss
                else:
                    losses['total'] = ref_loss
            
        # print(losses)
        if compute_loss:
            return task_outs, losses
        else:
            return task_outs

    def compute_instance_segmentation_loss(
        self, pred_obj_masks, pred_obj_conf_scores, pred_obj_img_fts, pred_obj_txt_fts,
        tgt_obj_masks, tgt_img_fts, tgt_obj_txt_fts
    ):
        batch_size = pred_obj_masks.size(0)
        device = pred_obj_masks.device
        pred_obj_masks_sigmoid = torch.sigmoid(pred_obj_masks)

        # compute cost functions for hungarian matching
        indices = []
        for b in range(batch_size):
            num_pred_objs = pred_obj_masks[b].size(0)
            num_tgt_objs = tgt_obj_masks[b].size(0)
            tgt_obj_masks_b = tgt_obj_masks[b].float()
            # bce costs: (num_pred_objs, npoints), (num_tgt_objs, npoints)
            # (num_pred_objs, num_tgt_objs, npoints) -> (num_pred_objs, num_tgt_objs)
            bce_costs = F.binary_cross_entropy_with_logits(
                pred_obj_masks[b].unsqueeze(1).expand(-1, num_tgt_objs, -1),
                tgt_obj_masks_b.unsqueeze(0).expand(num_pred_objs, -1, -1),
                reduction='none'
            ).mean(dim=-1)

            # dice costs: (num_pred_objs, npoints), (num_tgt_objs, npoints)
            # (num_pred_objs, num_tgt_objs)
            numerator = 2 * torch.einsum('np,mp->nm', pred_obj_masks_sigmoid[b], tgt_obj_masks_b)
            denominator = pred_obj_masks_sigmoid[b].sum(dim=-1).unsqueeze(1) + tgt_obj_masks_b.sum(dim=-1).unsqueeze(0)
            dice_costs = 1 - (numerator + 1) / (denominator + 1)

            # img and txt prediction costs
            img_clf_costs = F.smooth_l1_loss(
                pred_obj_img_fts[b].unsqueeze(1), tgt_img_fts[b].unsqueeze(0), 
                reduction='none'
            ).mean(dim=-1)
            txt_clf_costs = F.smooth_l1_loss(
                pred_obj_txt_fts[b].unsqueeze(1), tgt_obj_txt_fts[b].unsqueeze(0), 
                reduction='none'
            ).mean(dim=-1)

            conf_cost = 1 - torch.sigmoid(pred_obj_conf_scores[b])
            
            # combine costs
            # print(bce_costs, dice_costs, img_clf_costs, txt_clf_costs, conf_cost)
            costs = bce_costs + dice_costs + img_clf_costs + txt_clf_costs + conf_cost.unsqueeze(1)
            costs = costs.data.cpu()
            indices.append(linear_sum_assignment(costs))
        
        # compute losses given the bipartite matching
        losses = {
            'mask_bce': [], 'mask_dice': [],
            'conf': [], 'img_clf': [], 'txt_clf': [],
        }
        for b in range(batch_size):
            row_idx, col_idx = indices[b]

            # object mask loss
            pred_obj_masks_b = pred_obj_masks[b][row_idx]
            tgt_obj_masks_b = tgt_obj_masks[b][col_idx].float()
            mask_bce = F.binary_cross_entropy_with_logits(
                pred_obj_masks_b, tgt_obj_masks_b, reduction='mean'
            )
            losses['mask_bce'].append(mask_bce)

            numerator = 2 * (pred_obj_masks_sigmoid[b][row_idx] * tgt_obj_masks_b).sum(dim=-1)
            denominator = pred_obj_masks_sigmoid[b][row_idx].sum(dim=-1) + tgt_obj_masks_b.sum(dim=-1)
            mask_dice = 1 - (numerator + 1) / (denominator + 1)
            losses['mask_dice'].append(mask_dice.mean())

            # object confidence loss
            tgt_conf_scores = torch.zeros_like(pred_obj_conf_scores[b])
            tgt_conf_scores[row_idx] = 1
            conf_loss = F.binary_cross_entropy_with_logits(
                pred_obj_conf_scores[b], tgt_conf_scores, reduction='mean'
            )
            losses['conf'].append(conf_loss)

            # image and text classification loss
            losses['img_clf'].append(F.smooth_l1_loss(
                pred_obj_img_fts[b][row_idx], tgt_img_fts[b][col_idx], reduction='mean'
            ))
            losses['txt_clf'].append(F.smooth_l1_loss(
                pred_obj_txt_fts[b][row_idx], tgt_obj_txt_fts[b][col_idx], reduction='mean'
            ))
        
        for key, value in losses.items():
            losses[key] = torch.stack(value).mean()

        # losses['total'] = sum(list(losses.values()))
        losses['total'] = losses['mask_bce'] + losses['mask_dice'] + \
                          losses['conf'] + losses['img_clf'] + losses['txt_clf']
        return losses

    def forward_grasp(self, batch, compute_loss=False):
        '''batch data:
            pc_fts: (batch, npoints, dim)
            grasp_offsets: (batch, npoints, 3),
            grasp_rotations: (batch, npoints, 3, 3),
            grasp_valid_masks: (batch, npoints),
        '''
        batch = self.prepare_batch(batch)
        device = batch['pc_fts'].device
        batch_size = batch['pc_fts'].size(0)

        # encode point cloud
        mae_enc_outs = self.mae_encoder(
            batch['pc_fts'], mask_pc=False, return_ca_inputs=False,
        )
        pc_vis = mae_enc_outs['pc_vis']
        centers = mae_enc_outs['centers']   # (batch, num_groups, 3)

        enc_pc_poses = batch['pc_fts'][..., :3].contiguous()
        enc_pc_fts = three_interpolate_feature(
            enc_pc_poses, centers, pc_vis
        )   # (batch, npoints, hidden_size)

        grasp_outs = self.grasp_proj_head(enc_pc_fts)
        pred_logits = grasp_outs[..., 0]
        pred_offsets = grasp_outs[..., 1:4]
        pred_rotations = grasp_outs[..., 4:]

        pred_rot_matrices = GraspRotationMatrix.compute_rotation_matrix_from_ortho6d(
            pred_rotations.view(-1, 6)
        ).view(batch_size, -1, 3, 3)

        losses = {}
        losses['grasp_bce'] = F.binary_cross_entropy_with_logits(
            pred_logits, batch['grasp_valid_masks'].float(), reduction='mean'
        )
        losses['grasp_offsets'] = F.mse_loss(
            pred_offsets, batch['grasp_offsets'], reduction='none'
        )[batch['grasp_valid_masks'] == 1].mean()
        losses['grasp_rotations'] = F.mse_loss(
            pred_rot_matrices, batch['grasp_rotations'], reduction='none'
        )[batch['grasp_valid_masks'] == 1].mean()
        losses['total'] = losses['grasp_bce'] + \
                          losses['grasp_offsets'] + \
                          losses['grasp_rotations']
        
        task_outs = {
            'grasp_logits': pred_logits,
            'grasp_offsets': pred_offsets,
            'grasp_rotations': pred_rot_matrices,
        }        
            
        # print(losses)
        if compute_loss:
            return task_outs, losses
        else:
            return task_outs
        

class GraspRotationMatrix():
    # https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/shapenet/code/tools.py
    # batch*n
    @staticmethod
    def normalize_vector( v):
        batch=v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))# batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
        v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
        v = v/v_mag
        return v

    # u, v batch*n
    @staticmethod
    def cross_product( u, v):
        batch = u.shape[0]
        #print (u.shape)
        #print (v.shape)
        i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
        j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
        k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
            
        out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
            
        return out

    #poses batch*6
    @staticmethod
    def compute_rotation_matrix_from_ortho6d(poses):
        x_raw = poses[:,0:3]#batch*3
        y_raw = poses[:,3:6]#batch*3
            
        x = GraspRotationMatrix.normalize_vector(x_raw) #batch*3
        z = GraspRotationMatrix.cross_product(x,y_raw) #batch*3
        z = GraspRotationMatrix.normalize_vector(z)#batch*3
        y = GraspRotationMatrix.cross_product(z,x)#batch*3
            
        x = x.view(-1,3,1)
        y = y.view(-1,3,1)
        z = z.view(-1,3,1)
        matrix = torch.cat((x,y,z), 2) #batch*3*3
        return matrix