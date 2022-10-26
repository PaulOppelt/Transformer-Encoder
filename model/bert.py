import torch
import torch.nn as nn

from torch import Tensor

from .encoderblock import EncoderBlock
from .embedding import EmbeddingLayer

from typing import Optional


class Bert(nn.Module):
    r"""Implement Bert Network that can be pretrained on a reconstruction task and finetuned with labeled data
    Args: 
        d_model: embedding dimension of the input tokens
        n_segments: embeddings for segments in the input: e.g Sentences.
        vocab_size: number of tokens in the feature space
        n_layers: number of self-attention layers
        nheads: nuber of self-attention heads in the layers
        PAD_IDX: index in vocabulary to be a padding index. gradients w.r.t the pad-index are ignored. 
        copy_weight: bool, embedding weight is the same as the output weight of the reconstruction layer. 
    """
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

        # initialize EncoderBlock with init-parameters
        self.Block = EncoderBlock(
            d_model=d_model,
            nhead=nhead,
            d_hidden=d_hidden,
            n_layers=n_layers,
            dropout=dropout,
        )

        # initialize Embedding layeres
        self.Embedding = EmbeddingLayer(
            n_segment=n_segments,
            vocab_size=vocab_size,
            d_model=d_model,
            padding_idx=PAD_IDX,
            scale_grad_by_freq=True,
        )

        # define output layers.
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
        r"""
        Args: 
            src: Tensor, shape: batch, sequence_lenght

        out: 
            src: ouput of the Transformer, can be passed to linear-classification
                layer in order to predict labeled data in a finetuning step
                    shape: batch, sequence_lenght, d_model
            logits_te: reconstruction of the input. Still needs to be passed to a loss-function to predict
                most likely reconstruction. 
                    shape: batch, sequence_lenght, vocab_size
                
        """
        src = self.Embedding(src, segments, strain)
        src = self.Block(src)
        logits_te = self.norm(self.activation(self.lc(src)))
        logits_te = self.out(logits_te)
        logits_te = torch.permute(logits_te, (0, 2, 1))
        return logits_te, src
