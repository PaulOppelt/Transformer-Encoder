from model import Bert
from model import PositionalEncoding
import torch
import time

import torch
import torch.nn as nn
from torch import Tensor
import math
from torch.autograd import Variable

# count to total number of parameters in a model
def get_n_params(model):
    r"""function tp print the number of parameters in the model
    """
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


# create segments for the embedding layer
def segment_gen(n_segment: int, n_domains_incl: int, batch_size: int) -> Tensor:
    segments = torch.arange(n_segment)
    segments = segments.repeat(n_domains_incl)
    segments = segments.view(n_domains_incl, n_segment).T
    segments = segments.reshape(-1).repeat(batch_size)
    segments = segments.view(-1, n_segment * n_domains_incl)
    return segments


# test classification network
class LinearClassification(nn.Module):
    r"""
    description: missing. Move class to modules included with wrapper.
    """

    def __init__(
        self, d_model: int, n_domains_incl: int, window_size: int, n_labels: int
    ) -> None:
        super().__init__()

        self.n_domains_incl = n_domains_incl
        self.window_size = window_size
        self.d_model = d_model

        self.fc = nn.Linear(d_model, d_model * self.n_domains_incl)
        self.activ = torch.nn.functional.gelu
        self.drop = nn.Dropout(p=0.4)
        self.norm = nn.LayerNorm(d_model * self.n_domains_incl)
        self.classifier = nn.Linear(d_model * self.n_domains_incl, n_labels)

    def forward(self, input: "Tensor") -> "Tensor":
        input = input[:, :: self.n_domains_incl]
        pooled_h = self.norm(self.activ(self.fc(input)))  # [0]
        logits = self.classifier(pooled_h)
        logits = torch.permute(
            logits, (0, 2, 1)
        )  # [seq_lenght, batch, emb_size] -> [batch, emb_size, seq_lenght]
        return logits


if __name__ == "__main__":
    # torch.manual_seed(0)
    # test positional encoding:
    # pos = PositionalEncoding(d_model=512, dropout=0.1)
    # print(pos(torch.ones(32,2,512)))

    # test Encoder
    Encoder = Bert(
        vocab_size=100,
        n_domains_incl=4,
        window_size=32,
        d_model=512,
        n_layers=12,
        n_heads=8,
        feed_foreward=1024,
        PAD_IDX=1,
        dropout=0,
        scale_grad_by_freq=True,
        copy_weight=False,
    )

    # test classification network
    LinearClassification = LinearClassification(512, 4, 8, 2)

    input = torch.randint(5, 100, (1, 32)).to(torch.long)
    segments = segment_gen(8, 4, 1)

    logits, src = Encoder(input, segments)
    out = LinearClassification(src)
    loss = torch.nn.functional.cross_entropy(
        out, torch.randint(0, 1, (1, 8)).to(torch.long)
    )
    loss.backward()
    print(loss)

    m = nn.ReLU()
    encoder_layers = [
        Encoder.Block.Encoder_Layers[i].MultiHeadAttention.Attention.attention
        for i in range(12)
    ]
    res = torch.amax(encoder_layers[0], dim=1) * m(
        torch.amax(encoder_layers[0].grad, dim=1)
    )
    print(res.shape)
    for i in encoder_layers[1:]:
        # print(res)
        print(torch.amax(i, dim=1) * m(torch.amax(i.grad, dim=1)))
        # print(m(torch.mean(i.grad,dim=1)))
    # print(res)

 

    