import numpy as np

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_


class BaseModel(nn.Module):
    
    @property
    def num_parameters(self):
        nweights, nparams = 0, 0
        for k, v in self.named_parameters():
            nweights += np.prod(v.size())
            nparams += 1
        return nweights, nparams

    @property
    def num_trainable_parameters(self):
        nweights, nparams = 0, 0
        for k, v in self.named_parameters():
            if v.requires_grad:
                nweights += np.prod(v.size())
                nparams += 1
        return nweights, nparams

    def prepare_batch(self, batch):
        device = next(self.parameters()).device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            trunc_normal_(m.weight, std=.02)