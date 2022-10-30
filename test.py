from model import Bert
from model import PositionalEncoding
import torch
import time

import torch
import torch.nn as nn
from torch import Tensor
import math

def get_n_params(model):
    r"""function tp print the number of parameters in the model
    """
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def segment_gen(n_segment: int, n_domains_incl: int, batch_size: int) -> Tensor:
    segments = torch.arange(n_segment)
    segments = segments.repeat(n_domains_incl)
    segments = segments.view(n_domains_incl, n_segment).T
    segments = segments.reshape(-1).repeat(batch_size)
    segments = segments.view(-1, n_segment * n_domains_incl)
    return segments


if __name__ == "__main__": 
    # test positional encoding:
    pos = PositionalEncoding(d_model=512, dropout=0.1)
    print(pos(torch.ones(32,2,512)))



    # test Encoder
    Encoder = Bert(
        vocab_size = 100,
        n_domains_incl = 4,
        window_size = 32,
        d_model = 512,
        n_layers = 12,
        n_heads = 8,
        feed_foreward = 1024,
        PAD_IDX = 1,
        dropout = 0.4,
        scale_grad_by_freq = True,
        copy_weight =  False)

    print("number of parameters=",get_n_params(Encoder))

    input = torch.randint(0,100,(256,32)).to(torch.long)
    segments = segment_gen(8,4,256)

    start = time.time()
    print(Encoder(input, segments))
    print(time.time()-start)
