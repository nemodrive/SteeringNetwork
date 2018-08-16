import torch.nn as nn
import numpy as np


def weight_init_ortho(module):
    """
    Initializes Linear, Conv2d, and LSTM weights.
    """
    classname = module.__class__.__name__

    if classname == 'Linear':
        module.weight.data = nn.init.orthogonal(module.weight.data, gain=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'Conv2d':
        module.weight.data = nn.init.orthogonal(module.weight.data, gain=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'LSTM':
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                param.data = nn.init.orthogonal(param.data, gain=1.)
            if 'weight_hh' in name:
                param.data = nn.init.orthogonal(param.data, gain=1.)
            if 'bias' in name:
                param.data.zero_()
