import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class AttentionHead(nn.Module):
    def __init__(self, d_model: int):

        super(AttentionHead, self).__init__()

        self.attention = None

    def forward(self, Q, K, V):
        r"""
         Args:
             Q, K, V: Tensor, shape: batch, nheads, seq_lenght, d_model
             
         Transpose the last two dimensions in order to perform a dot product between
         all Input Embeddings.
         
         out: 
             Attention: Tensor, shape: batch, nheads, seq_lenght, d_model
                 = Softmax(Q @ K.T / sqrt(d_k)) @ V, d_k = d_model // nheads
             
             if single Head Attention: d_k = d_model
        """
        self.attention =  Variable(nn.Softmax(dim=-1)((Q @ K.transpose(-1, -2)) / math.sqrt(Q.shape[-1])), requires_grad=True)
        # store attention value for rollout.

        return (
            self.attention @ V
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, nhead: int, d_model: int):

        super(MultiHeadAttention, self).__init__()

        assert (
            d_model % nhead == 0
        ), "Embedding dimension must be devisible by number of self attention heads"

        self.nhead = nhead
        self.d_model = d_model
        self.d_k = d_model // nhead

        self.Attention = AttentionHead(d_model=d_model)

        """
        define the query, key and value matrix which cast the input vecor to the n different
        self Attention heads
        """
        self.W_Q = nn.Linear(d_model, self.d_model, bias=True)
        self.W_V = nn.Linear(d_model, self.d_model, bias=True)
        self.W_K = nn.Linear(d_model, self.d_model, bias=True)

        self.W_O = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        r"""
        Args: x, shape batch, sequence_lenght, d_model. 

        In order to perform multihead attention we create three linear projection matrices which are seperated into
        nhead- parts and then individually perform self-Attention with the input vector x.

        MultiHeadAttention = concat(head_1, head_2, ..., head_n) @ W_0

        head_i = Attention(Q @ W_q_i,V @ W_v_i, K @ W_k_i), where the submatrices can be created by seperating large matrices with
        dimensionality : d_model x d_model.

        Attention is performed accross the last two dimensions of the input queries, keys and values:
            nhead x d_k

        out:
             shape: batch, seq_lenght, d_model
        """
        batch_size = x.shape[0]
        q = self.W_Q(x).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        k = self.W_K(x).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        v = self.W_V(x).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        return self.W_O(
            self.Attention(q, k, v)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )
