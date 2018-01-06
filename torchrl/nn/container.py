import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from torchrl.utils import get_obj


class SequentialExtended(nn.Module):
    def __init__(self, layers_config):
        super().__init__()
        self.layers = nn.Sequential(layers_config)

    def _maybe_cuda(self, x):
        return x.cuda() if self.is_cuda else x

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = Variable(self._maybe_cuda(torch.from_numpy(x).float()))
        output = self.layers(x)

        return output

    def get_output_shape(self, input_shape):
        fake_input = Variable(self._maybe_cuda(torch.zeros(input_shape)[None]))
        out = self.layers(fake_input)
        shape = out.shape[1:]

        return torch.IntTensor(list(shape))

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    @classmethod
    def from_config(cls, config, kwargs):
        layers_config = OrderedDict(
            {key: get_obj(value)
             for key, value in config.items()})

        return cls(layers_config, **kwargs)
