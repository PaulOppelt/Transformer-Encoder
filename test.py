from model import Bert
import torch
import time

import torch
import torch.nn as nn
from torch import Tensor
import math
from functions import segment_gen, get_n_params
from model import LinearClassification


if __name__ == "__main__": 
    # add manual seed for reproducibility
    torch.manual_seed(0)

    # initialize bert model
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

    # initialize downstream classification network
    LinearClassification = LinearClassification(512,4,8,2)

    batch_size = 32

    input = torch.randint(0,100,(batch_size, 32)).to(torch.long)
    print(input.shape)
    segments = segment_gen(8,4,32)

    start = time.time()
    logits, src = Encoder(input, segments)
    out = LinearClassification(src)
    print(time.time()-start)




      




