import torch
import torch.nn as nn

from encoderlayer import EncoderLayer


class EncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_hidden, n_layers, dropout):

        super(EncoderBlock, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.d_hidden = d_hidden
        self.dropout = dropout

        self.Encoder_Layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    d_hidden=self.d_hidden,
                    dropout=self.dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        r"""
        pass the input through n_layers of multihead Attention with subsequent
        positionwise feedforward
        """
        for layer in self.Encoder_Layers:
            x = layer(x)

        return x
