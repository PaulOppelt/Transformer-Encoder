import torch
import torch.nn as nn

from torch import Tensor
import math


class PositionalEncoding(nn.Module):
    r"""Encodes the input of a sequence such, that the model can learn recognize relative position
    in the input. Otherwise the input is incariant to positional permutation.
    Args:
        d_model: embedding dimension of the input tokens
    """
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
		Args:
			x: Tensor, shape [batch_size,seq_len, d_model]
		"""
        return self.pe[:, :x.size(1)]


class PositionwiseFeedForward(nn.Module):
    r"""perform a positionwise feedforward operation on the input. Each input embedding vector is multiplied
    with the same weight matrix. 
    """
    def __init__(self, d_model: int, d_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
