import torch
import torch.nn as nn

from . import PositionwiseFeedForward
from . import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_hidden, dropout):

        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.d_hidden = d_hidden

        self.PosWiseFeedForeward = PositionwiseFeedForward(
            d_model=self.d_model, d_hidden=self.d_hidden, dropout=dropout
        )
        self.MultiHeadAttention = MultiHeadAttention(
            nhead=self.nhead, d_model=self.d_model
        )

        self.norm_attention = nn.LayerNorm(self.d_model)
        self.norm_pos = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.MultiHeadAttention(x)
        out = self.dropout(self.norm_attention(attention + x))
        feed_forward = self.PosWiseFeedForeward(out)
        out = self.dropout(self.norm_pos(feed_forward + out))
        return out
