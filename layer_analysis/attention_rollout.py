import torch
import torch.nn as nn


def rollout(model):
    r"""
    Args:
        model: torch.nn.Module
    """
    activation = nn.ReLU()
    encoder_layers = [
        model.Block.Encoder_Layers[i].MultiHeadAttention.Attention.attention
        for i in range(model.block.n_layers)
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


class SaveOutput:
    r"""class to store the gradients of an arbitraty layer within a neuronal network using
    bachward hooks
    """
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

