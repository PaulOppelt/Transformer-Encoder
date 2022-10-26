import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        n_segment: int,
        vocab_size: int,
        d_model: int,
        padding_idx: int,
        scale_grad_by_freq: bool,
    ):

        super(EmbeddingLayer, self).__init__()
        r"""
        The Input to the self-attention passes through three different Embedding layers:
            TokenEmbedding: Encodes every individual token in the input with a n_model- dimensional vector. Tokens
                are represented by one-hot vectors before passing into the token- embedding layer.
            
            SegmentEmbedding: Each segment is represented by a learned vector. E.g an input vector consisting of 10 genes,
                every gene is represented by a Segment vector that is added to the each token vecor in the gene. 

            StrandEmbedding: Optional: directionality of the gene-reading-direction encoded into a vector that is added.
        """

        self.TokenEmbedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=None,
            device=None,
            dtype=None,
        )
        self.SegmentEmbedding = nn.Embedding(
            num_embeddings=n_segment,
            embedding_dim=d_model,
            padding_idx=None,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=None,
            device=None,
            dtype=None,
        )
        self.StrandEmbedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=d_model,
            padding_idx=None,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=None,
            device=None,
            dtype=None,
        )

    def forward(self, x, segment=None, strand=None):
        if segment == None and strand == None:
            segment = torch.zeros(x.shape).to(torch.long)
            strand = torch.zeros(x.shape).to(torch.long)
        return (
            self.TokenEmbedding(x)
            + self.SegmentEmbedding(segment)
            + self.StrandEmbedding(strand)
        )
