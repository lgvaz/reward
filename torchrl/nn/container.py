from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from torchrl.utils import get_obj


class ModuleExtended(nn.Module):
    def _maybe_cuda(self, x):
        return x.cuda() if self.is_cuda and not x.is_cuda else x

    def _to_variable(self, x):
        if isinstance(x, np.ndarray):
            # pytorch doesn't support bool
            if x.dtype == 'bool':
                x = x.astype('int')
            # we want only single precision floats
            if x.dtype == 'float64':
                x = x.astype('float32')

            x = torch.from_numpy(x)

        if not isinstance(x, Variable):
            x = Variable(x)

        return self._maybe_cuda(x)

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def get_output_shape(self, input_shape):
        fake_input = Variable(self._maybe_cuda(torch.zeros(input_shape)[None]))
        out = self.layers(fake_input)
        shape = out.shape[1:]

        return torch.IntTensor(list(shape))


class SequentialExtended(ModuleExtended):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(*args, **kwargs)

    def forward(self, x):
        return self.layers(self._to_variable(x))

    @classmethod
    def from_config(cls, config, kwargs):
        layers_config = OrderedDict(
            {key: get_obj(value)
             for key, value in config.items()})

        return cls(layers_config, **kwargs)


class FlattenLinear(nn.Linear):
    def __init__(self, in_features, out_features, **kwargs):
        if isinstance(in_features, torch.IntTensor):
            in_features = in_features.prod()
        else:
            in_features = int(np.prod(in_features))

        super().__init__(in_features=in_features, out_features=out_features, **kwargs)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        return super().forward(x)


class ActionLinear(FlattenLinear):
    pass
