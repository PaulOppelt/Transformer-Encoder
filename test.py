from model import Bert
from model import PositionalEncoding
import torch
import time

import torch
import torch.nn as nn
from torch import Tensor
import math
from functions import segment_gen, get_n_params
from model import LinearClassification


if __name__ == "__main__": 
    torch.manual_seed(0)
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

    LinearClassification = LinearClassification(512,4,8,2)

    print("number of parameters=",get_n_params(Encoder))

    input = torch.randint(0,100,(32,32)).to(torch.long)
    segments = segment_gen(8,4,32)

    start = time.time()
    logits, src = Encoder(input, segments)
    out = LinearClassification(src)
    print(time.time()-start)





