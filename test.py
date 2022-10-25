from model import Bert
import torch


if __name__ == "__main__": 
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

    print(Encoder)

    Encoder(torch.randint(0,100,(256,32)).to(torch.long))
