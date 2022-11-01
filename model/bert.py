import torch
import torch.nn as nn

from torch import Tensor

from .encoderblock import EncoderBlock
from .embedding import EmbeddingLayer
from .modules import PositionalEncoding

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
        vocab_size: int = 100,
        n_domains_incl: int = 4, # not used 
        window_size: int = 32,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 16,
        feed_foreward: int = 1024,
        PAD_IDX: int = 1,
        dropout: float = 0.4,
        scale_grad_by_freq: bool = True,
        copy_weight: bool = False,
        **kwargs
    ):

        super(Bert, self).__init__()

        # initialize EncoderBlock with init-parameters
        self.Block = EncoderBlock(
            d_model=d_model,
            nhead=n_heads,
            d_hidden=feed_foreward,
            n_layers=n_layers,
            dropout=dropout,
        )

        # initialize Embedding layeres
        self.Embedding = EmbeddingLayer(
            window_size=window_size,
            vocab_size=vocab_size,
            d_model=d_model,
            padding_idx=PAD_IDX,
            scale_grad_by_freq=scale_grad_by_freq
        )

        # initialize positional encoding to the input tokens.
        self.Positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)

        # define output layers.
        self.lc = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        self.out = nn.Linear(d_model, vocab_size)

        # copy the token-embedding weight to the output layer
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
        src = self.Positional_encoding(src)
        src = self.Block(src)
        logits_te = self.norm(self.activation(self.lc(src)))
        logits_te = self.out(logits_te) # depending on the loss function perform permutation -> batch, vocab, seq_lenght
        return logits_te, src 
