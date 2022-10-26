import torch
import torch.nn as nn

from .modules import PositionwiseFeedForward
from .attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    r"""This class implements one layer of the Transformer Encoder. I consists of a Multihead-
    Attention layer followed by a positionwise feedforward layer. Both Layers have residual connections and
    subsequent layernormalization of the attention/ positionwise-feedforward - and the residual ouput.

    Args:
        d_model: Embedding dimension of the input tokens
        nhead: number of self-attention heads
        d_hidden: dimension of the positionwise feedforward layer
        dropout: probability to drop input to avoid overfitting
    """
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
        r"""
        Args: 
            x: Tensor, shape: batch, sequence_lenght, d_model
        out:
             shape: conserved
        """
        attention = self.MultiHeadAttention(x)
        out = self.dropout(self.norm_attention(attention + x))
        feed_forward = self.PosWiseFeedForeward(out)
        out = self.dropout(self.norm_pos(feed_forward + out))
        return out
