import torch

def get_n_params(model):
    r"""function tp print the number of parameters in the model
    Args:
        model: torch.nn.Module
    returns:
        number of parameters
    """
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def segment_gen(n_segment: int, n_domains_incl: int, batch_size: int) -> Tensor:
    r"""create segments for an embedding layer. Eeach feature in a segment gets the same encoding.
    Args:
        n_segment: number of segments
        n_domains_incl: number of domains in the input
        batch_size: batch size  
    Returns: 
        segments: tensor of shape (batch_size, n_segment * n_domains_incl)
    """
    segments = torch.arange(n_segment)
    segments = segments.repeat(n_domains_incl)
    segments = segments.view(n_domains_incl, n_segment).T
    segments = segments.reshape(-1).repeat(batch_size)
    segments = segments.view(-1, n_segment * n_domains_incl)
    return segments