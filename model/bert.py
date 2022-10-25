import torch
import torch.nn as nn

from torch import Tensor

from .encoderblock import EncoderBlock
from .embedding import EmbeddingLayer

from typing import Optional


class Bert(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_segments: int = 32,
        vocab_size: int = 100,
        n_layers: int = 12,
        nhead: int = 8,
        d_hidden: int = 1024,
        PAD_IDX: int = 1,
        dropout: float = 0.1,
        copy_weight: bool = False,
    ):

        super(Bert, self).__init__()

        self.Block = EncoderBlock(
            d_model=d_model,
            nhead=nhead,
            d_hidden=d_hidden,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.Embedding = EmbeddingLayer(
            n_segment=n_segments,
            vocab_size=vocab_size,
            d_model=d_model,
            padding_idx=PAD_IDX,
            scale_grad_by_freq=True,
        )

        self.lc = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        self.out = nn.Linear(d_model, vocab_size)

        if copy_weight == True:
            weights = self.Embedding.weight
            self.out.weight = weights

    def forward(
        self,
        src: Tensor,
        segments: Optional[None] = None,
        strain: Optional[None] = None,
    ) -> Tensor:
        src = self.Embedding(src, segments, strain)
        src = self.Block(src)
        logits_te = self.norm(self.activation(self.lc(src)))
        logits_te = self.out(logits_te)
        logits_te = torch.permute(logits_te, (0, 2, 1))
        return logits_te, src
