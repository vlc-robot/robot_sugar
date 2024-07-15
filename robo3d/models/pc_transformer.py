import random
import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


from .transformer import SelfAttentionBlock, CrossAttentionBlock
from .point_ops import PointGroup


class PCGroupEncoder(nn.Module):  # Embedding module
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.first_conv = nn.Sequential(
            nn.Conv1d(self.input_size, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.output_size, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, dim = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, dim)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # (b*g, 256, n)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (b*g, 256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # (b*g, 512, n)
        feature = self.second_conv(feature)  # (b*g, output_size, n)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # (b*g, output_size)
        return feature_global.reshape(bs, g, self.output_size)


class Block(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
        drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
        cross_attn_input='post', reproduce_recon=False
    ):
        super().__init__()
        self.reproduce_recon = reproduce_recon
        if self.reproduce_recon:
            self.cross_attn_input = 'extra'
        else:
            self.cross_attn_input = cross_attn_input
        assert self.cross_attn_input in ['pre', 'post', 'extra']

        self.encoder_block = SelfAttentionBlock(
            dim, num_heads, mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, qk_scale=qk_scale, 
            drop=drop, attn_drop=attn_drop, drop_path=drop_path, 
            act_layer=act_layer, norm_layer=norm_layer, 
            reproduce_recon=reproduce_recon
        )

        self.decoder_block_sa = SelfAttentionBlock(
            dim, num_heads, mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, qk_scale=qk_scale, 
            drop=drop, attn_drop=attn_drop, drop_path=drop_path, 
            act_layer=act_layer, norm_layer=norm_layer, 
            reproduce_recon=reproduce_recon
        )
        self.decoder_block_ca = CrossAttentionBlock(
            dim, num_heads, mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, qk_scale=qk_scale, 
            drop=drop, attn_drop=attn_drop, drop_path=drop_path, 
            act_layer=act_layer, norm_layer=norm_layer, 
            reproduce_recon=reproduce_recon
        )

    def update_tgt_given_src(
        self, src, tgt, src_padded_mask=None, tgt_padded_mask=None, 
        skip_tgt_sa=False, detach_src=False
    ):
        if skip_tgt_sa:
            tgt_out = tgt
        else:
            tgt_out = self.decoder_block_sa(tgt, tgt_padded_mask)

        tgt_out = self.decoder_block_ca(
            tgt_out, src, src_padded_mask=src_padded_mask, detach_src=detach_src
        )
        return tgt_out

    def forward(
        self, src, tgt=None, src_padded_mask=None, tgt_padded_mask=None, 
        skip_tgt_sa=False, detach_src=False
    ):
        src_out = self.encoder_block(
            src, padded_mask=src_padded_mask
        )
        if self.reproduce_recon:
            src_out, src_extra_out = src_out

        if self.cross_attn_input == 'post':
            cross_attn_input = src_out
        elif self.cross_attn_input == 'pre':
            cross_attn_input = src
        elif self.cross_attn_input == 'extra':
            cross_attn_input = src_extra_out['norm1']

        if tgt is None:
            return src_out, None, cross_attn_input
        
        if skip_tgt_sa:
            tgt_out = tgt
        else:
            tgt_out = self.decoder_block_sa(tgt, tgt_padded_mask)

        tgt_out = self.decoder_block_ca(
            tgt_out, cross_attn_input, src_padded_mask=src_padded_mask, detach_src=detach_src
        )
        return src_out, tgt_out, cross_attn_input


class MaskedPCTransformer(nn.Module):
    def __init__(
        self, hidden_size=384, num_heads=6, depth=12, 
        input_size=6, num_groups=64, group_size=32, 
        group_use_knn=True, group_radius=None,
        drop_path_rate=0.1, mask_ratio=0., mask_type='rand', 
        cross_attn_input='post', cross_attn_layers=None, **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.input_size = input_size
        self.num_groups = num_groups
        self.group_size = group_size
        self.group_use_knn = group_use_knn
        self.group_radius = group_radius
        self.drop_path_rate = drop_path_rate
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.cross_attn_input = cross_attn_input
        self.cross_attn_layers = cross_attn_layers
        self.kwargs = kwargs

        # embedding
        self.group_divider = PointGroup(
            num_groups=self.num_groups, group_size=self.group_size, 
            knn=self.group_use_knn, radius=self.group_radius
        )
        self.encoder = PCGroupEncoder(self.input_size, self.hidden_size)

        self.point_pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.hidden_size),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=self.hidden_size, num_heads=self.num_heads, mlp_ratio=4, 
                qkv_bias=False, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i],
                cross_attn_input=self.cross_attn_input
            )
            for i in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.hidden_size)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G
    
    def update_query_given_pc_layer_outs(
        self, pc_layer_outs, query, query_pos=None, query_padded_mask=None,
        skip_tgt_sa=False, detach_src=False,
    ):
        '''
            pc_layer_outs: [(batch, num_groups, dim)],
            query: (batch, ntokens, dim)
        '''
        for kth, block in enumerate(self.blocks):
            if query_pos is not None:
                query = query + query_pos

            query = block.update_tgt_given_src(
                pc_layer_outs[kth], query, tgt_padded_mask=query_padded_mask, 
                skip_tgt_sa=skip_tgt_sa, detach_src=detach_src
            )
        
        query = self.norm(query)
        return query

    def forward(
        self, pc_fts, query=None, query_pos=None, query_padded_mask=None, 
        skip_tgt_sa=False, detach_src=False, return_multiscale_layers=None,
        return_ca_inputs=False, mask_pc=True,
    ):
        '''
            pc_fts : (batch, npoints, dim), the first 3 dimensions are the xyz coordinates
            query: (batch, ntokens, dim)
        '''
        neighborhoods, centers = self.group_divider(pc_fts)

        # generate mask
        if mask_pc:
            if self.mask_type == 'rand':
                pc_bool_masks = self._mask_center_rand(centers, noaug=False)  # B G
            else:
                pc_bool_masks = self._mask_center_block(centers, noaug=False)
        else:
            pc_bool_masks = torch.zeros(centers.shape[:2]).bool().to(centers.device)

        group_input_tokens = self.encoder(neighborhoods)  # B G C
        batch_size, seq_len, C = group_input_tokens.size()

        pc_vis = group_input_tokens[~pc_bool_masks].reshape(batch_size, -1, C)
        # add pos embedding
        centers_vis = centers[~pc_bool_masks].reshape(batch_size, -1, 3)
        pos_vis = self.point_pos_embedding(centers_vis)

        # transformer
        if return_multiscale_layers is not None:
            multiscale_pc_fts = []
        if return_ca_inputs:
            ca_inputs = []

        for kth, block in enumerate(self.blocks):
            if query is not None and query_pos is not None:
                query = query + query_pos

            pc_vis, query, ca_input = block(
                pc_vis + pos_vis, tgt=query, tgt_padded_mask=query_padded_mask,
                skip_tgt_sa=skip_tgt_sa, detach_src=detach_src
            )
            if return_ca_inputs:
                ca_inputs.append(ca_input)

            if kth == len(self.blocks) - 1:
                pc_vis = self.norm(pc_vis)
                if query is not None:
                    query = self.norm(query)

            if (return_multiscale_layers is not None) and ((kth + 1) in return_multiscale_layers):
                multiscale_pc_fts.append(pc_vis)

        outs = {
            'pc_vis': pc_vis,
            'query': query,
            'centers': centers,
            'neighborhoods': neighborhoods,
            'pc_bool_masks': pc_bool_masks,
        }
        if return_multiscale_layers is not None:
            outs['multiscale_pc_fts'] = torch.cat(multiscale_pc_fts, dim=2)
        if return_ca_inputs:
            outs['ca_inputs'] = ca_inputs
            if self.cross_attn_layers is not None:
                outs['ca_inputs'] = [outs['ca_inputs'][i] for i in self.cross_attn_layers]
        return outs
