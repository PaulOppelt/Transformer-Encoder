import torch
import torch.nn as nn

from .encoderlayer import EncoderLayer


class EncoderBlock(nn.Module):
    r"""class that stacks several Encoder layers.
    Args:
        n_layers: number of Encoder Layers
    """
    def __init__(self, d_model, nhead, d_hidden, n_layers, dropout):

        super(EncoderBlock, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.n_layers = n_layers

        self.Encoder_Layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    d_hidden=self.d_hidden,
                    dropout=self.dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(self, x):
        for layer in self.Encoder_Layers:
            x = layer(x)

        return x
