import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_

from .base import BaseModel
from .pc_transformer import MaskedPCTransformer


class PCTClassification(BaseModel):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = config.transformer_config.hidden_size
        self.cls_type = config.cls_type
        self.num_classes = config.num_classes

        self.mae_encoder = MaskedPCTransformer(**config.transformer_config)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.img_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.img_pos = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.txt_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.txt_pos = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        if self.cls_type == "linear":
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.hidden_size * 4, self.num_classes)
            )
        else:
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.hidden_size * 4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.num_classes)
            )

        if self.cls_type != 'full':
            for param_name, param in self.named_parameters():
                if not (param_name.startswith('cls_head_finetune') \
                        or param_name.startswith('cls_token') \
                        or param_name.startswith('cls_pos')):
                    param.requires_grad = False

        self.build_loss_func()

        self.apply(self._init_weights)
        for token in [self.cls_token, self.cls_pos, self.img_token, self.img_pos, self.txt_token, self.txt_pos]:
            trunc_normal_(token, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def forward(self, batch, compute_loss=False):
        '''batch data:
            pc_fts: (batch, npoints, dim)
        '''
        batch = self.prepare_batch(batch)
        batch_size = batch['pc_fts'].size(0)
        
        # encode point cloud
        mae_enc_outs = self.mae_encoder(batch['pc_fts'], return_ca_inputs=True)

        pc_fts = mae_enc_outs['pc_vis'] # (batch, npoints, dim)
        pc_layer_outs = mae_enc_outs['ca_inputs']

        cls_tokens = self.cls_token.expand(batch_size, 1, -1)
        cls_pos_embeds = self.cls_pos.expand(batch_size, 1, -1)
        img_tokens = self.img_token.expand(batch_size, 1, -1)
        img_pos_embeds = self.img_pos.expand(batch_size, 1, -1)
        txt_tokens = self.txt_token.expand(batch_size, 1, -1)
        txt_pos_embeds = self.txt_pos.expand(batch_size, 1, -1)

        query_tokens = torch.cat([cls_tokens, img_tokens, txt_tokens], dim=1)
        query_pos_embeds = torch.cat([cls_pos_embeds, img_pos_embeds, txt_pos_embeds], dim=1)

        pc_query_outs = self.mae_encoder.update_query_given_pc_layer_outs(
            pc_layer_outs, query_tokens, query_pos_embeds,
            query_padded_mask=None, 
            skip_tgt_sa=self.config.transformer_config.get('csc_skip_dec_sa', False), 
            detach_src=self.config.transformer_config.get('detach_enc_dec', False)
        )
        
        concat_fts = torch.cat([pc_query_outs.view(batch_size, -1), pc_fts.max(1)[0]], dim=1)
        logits = self.cls_head_finetune(concat_fts)

        if compute_loss:
            losses = {}
            losses['total'] = self.loss_ce(logits, batch['labels'].long())
            return logits, losses
            
        return logits

