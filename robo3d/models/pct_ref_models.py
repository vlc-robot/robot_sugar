import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from timm.models.layers import trunc_normal_

from .base import BaseModel
from .pc_transformer import MaskedPCTransformer
from .point_ops import three_interpolate_feature


class PCTRefModel(BaseModel):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = config.transformer_config.hidden_size

        ref_cfg = config.ref_decoder_config
        self.multiscale_pc_layers = ref_cfg.multiscale_pc_layers
        if len(self.multiscale_pc_layers) > 1:
            self.multiscale_fusion = nn.Linear(
                self.hidden_size * len(self.multiscale_pc_layers), self.hidden_size
            )
        else:
            self.multiscale_fusion = None

        self.mae_encoder = MaskedPCTransformer(**config.transformer_config)

        self.ref_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.ref_pos = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        if not config.freeze_all_except_head:
            self.ref_proj_head = nn.Linear(self.hidden_size, self.hidden_size)
        else:
            self.ref_proj_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
        
        self.ref_txt_fc = nn.Linear(ref_cfg.txt_ft_size, self.hidden_size)
        self.ref_txt_pos = nn.Parameter(torch.zeros(1, ref_cfg.max_txt_len, self.hidden_size))
        
        self.apply(self._init_weights)
        for token in [self.ref_token, self.ref_pos, self.ref_txt_pos]:
            trunc_normal_(token, std=.02)

    def forward(self, batch, compute_loss=False):
        '''batch data:
            pc_fts: (batch, npoints, dim)
            txt_embeds: (batch, txt_dim)
        '''
        batch = self.prepare_batch(batch)
        device = batch['pc_fts'].device
        batch_size = batch['pc_fts'].size(0)

        # encode point cloud
        mae_enc_outs = self.mae_encoder(
            batch['pc_fts'], mask_pc=False, return_ca_inputs=True,
            return_multiscale_layers=self.multiscale_pc_layers
        )
        pc_vis = mae_enc_outs['multiscale_pc_fts']
        if self.multiscale_fusion is not None:
            pc_vis = self.multiscale_fusion(pc_vis)
        centers = mae_enc_outs['centers']   # (batch, num_groups, 3)

        enc_pc_poses = batch['pc_fts'][..., :3].contiguous()
        enc_pc_fts = three_interpolate_feature(
            enc_pc_poses, centers, pc_vis
        )

        # object detection decoder
        pc_layer_outs = mae_enc_outs['ca_inputs']
        ref_token = self.ref_token.expand(batch_size, -1, -1)
        ref_pos_embed = self.ref_pos.expand(batch_size, -1, -1)
        ref_txt_tokens = self.ref_txt_fc(batch['txt_embeds'])
        max_txt_len = ref_txt_tokens.size(1)
        ref_txt_pos_embed = self.ref_txt_pos[:, :max_txt_len].expand(batch_size, -1, -1)

        query_tokens = torch.cat([ref_token, ref_txt_tokens], dim=1)
        query_pos_embeds = torch.cat([ref_pos_embed, ref_txt_pos_embed], dim=1)
        query_padded_mask = torch.zeros(batch_size, query_tokens.size(1), dtype=torch.bool)
        for i, txt_len in enumerate(batch['txt_lens']):
            query_padded_mask[i, txt_len+1:] = True
        query_padded_mask = query_padded_mask.to(device)

        query_outs = self.mae_encoder.update_query_given_pc_layer_outs(
            pc_layer_outs, query_tokens, query_pos_embeds,
            query_padded_mask=query_padded_mask, skip_tgt_sa=False, 
            detach_src=self.config.transformer_config.detach_enc_dec
        )
        query_outs = self.ref_proj_head(query_outs[:, 0])

        pred_masks = torch.einsum(
            'bd,bpd->bp', query_outs, enc_pc_fts
        )
        
        if compute_loss:
            losses = {}
            loss_cfg = self.config.loss_config

            pc_labels = batch['pc_labels'].float()
            if loss_cfg.focal_loss:
                losses['ref_bce'] = torchvision.ops.sigmoid_focal_loss(
                    pred_masks, pc_labels.float(), alpha=0.25, gamma=2, reduction='mean'
                )
            else:
                losses['ref_bce'] = F.binary_cross_entropy_with_logits(
                    pred_masks, pc_labels, reduction='mean'
                )

            pred_masks_sigmoid = torch.sigmoid(pred_masks)
            numerator = 2 * (pred_masks_sigmoid * pc_labels).sum(dim=-1)
            denominator = pred_masks_sigmoid.sum(dim=-1) + pc_labels.sum(dim=-1)
            mask_dice = 1 - (numerator + 1) / (denominator + 1)
            losses['ref_dice'] = mask_dice.mean()

            losses['total'] = loss_cfg.bce_loss_weight * losses['ref_bce'] + \
                              loss_cfg.dice_loss_weight * losses['ref_dice']
            
        # print(losses)
        if compute_loss:
            return pred_masks, losses
        else:
            return pred_masks

    