from __future__ import division

import torch
from torch import nn
from torch.utils import data
from torch.utils.data.dataloader import default_collate
from torch.distributions.normal import Normal

import numpy as np
import collections

# Based on Pytorch default_collate
class Collate:
    def __init__(self, float_type):
        if float_type == "half":
           self.float_type = torch.float16
        elif float_type == "float":
           self.float_type = torch.float32
        elif float_type == "double":
           self.float_type = torch.float64
        else:
            raise ValueError("Invalid float_type.")

    def __call__(self, batch):
        batch = default_collate(batch)
        return self.post_default_collate(batch)

    def post_default_collate(self, batch):
        if isinstance(batch, torch.Tensor):
            if batch.dtype.is_floating_point:
                if batch.dtype != self.float_type:
                    return torch.tensor(batch, dtype=self.float_type)
            elif batch.dtype != torch.int64:
                return torch.tensor(batch, dtype=torch.int64)
            return batch
        elif isinstance(batch, collections.Mapping):
            return {key: self.post_default_collate(batch[key])
                for key in batch}
        elif isinstance(batch, collections.Sequence):
            return [self.post_default_collate(elem) for elem in batch]
        else:
            return batch

def loss_function(output, inputv):
    mu_dec, logvar_dec, mu_enc, logvar_enc = output

    dist = Normal(mu_dec, (0.5*logvar_dec).exp(), True)
    BCE = - dist.log_prob(inputv)

    KLD = torch.sum(1 + logvar_enc - mu_enc.pow(2) - logvar_enc.exp())
    KLD = -0.5 * KLD

    return (BCE + KLD).mean()
