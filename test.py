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

if __name__ == "__main__": 
    # test positional encoding:
    pos = PositionalEncoding(d_model=512, dropout=0.1)
    print(pos(torch.ones(32,2,512)))

    # test Encoder
    Encoder = Bert(
        d_model = 512,
        n_segments = 32,
        vocab_size = 100,
        n_layers = 12,
        nhead = 8,
        d_hidden = 1024,
        PAD_IDX = 1,
        dropout = 0.1,
        copy_weight = False)

    print("number of parameters=",get_n_params(Encoder))

    start = time.time()
    print(Encoder(torch.randint(0,100,(256,32)).to(torch.long)))
    print(time.time()-start)
