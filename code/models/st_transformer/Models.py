import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




def get_subsequent_mask(seq,dim=2):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()[:2] #batch_size, batch_length
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s,dim), device=seq.device, dtype=torch.uint8), diagonal=1) #[dim,length,length]
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1,-1)  # batch x dim x ls x ls
    return subsequent_mask


def get_square_subsequent_mask(seq,dim=2):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()[:2] #batch_size, batch_length
    mask = (torch.triu(torch.ones(len_s, len_s)) == 1).transpose(0, 1) #[len_s,len_s]
    #mask = mask.unsqueeze(0).expand(sz_b, -1, -1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)
        self.relu = nn.ReLU()

    def forward(self, data, non_pad_mask):
        out  = self.relu(data)
        out = self.linear(data)
        out = out * non_pad_mask
        return out

