from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from torchrl.utils import get_obj, to_tensor


class ModuleExtended(nn.Module):
    """
    A torch module with added functionalities.
    """

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def maybe_cuda(self, x):
        """
        Convert input tensor to cuda if available.

        Parameters
        ----------
        x: torch.Tensor
            A pytorch tensor.

        Returns
        -------
        torch.Tensor
            A pytorch tensor.
        """
        return x.cuda() if self.is_cuda else x

    def to_tensor(self, x):
        return to_tensor(x, cuda_default=self.is_cuda)

    def get_output_shape(self, input_shape):
        """
        Feed forward the current module to find out the output shape.

        Parameters
        ----------
        input_shape: list
            The input dimensions.

        Returns
        -------
        torch.IntTensor
            The dimensions of the output.
        """
        self.maybe_cuda(self.layers)
        fake_input = self.maybe_cuda(torch.zeros(input_shape)[None])
        out = self.layers(fake_input)
        shape = out.shape[1:]

        return torch.IntTensor(list(shape))


class SequentialExtended(ModuleExtended):
    """
    A torch sequential module with added functionalities.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(*args, **kwargs)

    def append(self, module, name=None):
        name = str(name or len(self.layers))
        self.layers.add_module(name=name, module=module)

    def forward(self, x):
        if len(self.layers) > 0:
            return self.layers(x)
        else:
            return x

    @classmethod
    def from_config(cls, config, kwargs):
        layers_config = OrderedDict(
            {key: get_obj(value) for key, value in config.items()}
        )

        return cls(layers_config, **kwargs)


class Flatten(nn.Module):
    """
    Flatten the input and apply a linear layer.
    """

    def forward(self, x):
        return x.view(x.shape[0], -1)


class FlattenLinear(nn.Linear):
    """
    Flatten the input and apply a linear layer.

    Parameters
    ----------
    in_features: list
        Size of each input sample.
    out_features: list
        Size of each output sample.
    """

    def __init__(self, in_features, out_features, **kwargs):
        if isinstance(in_features, torch.IntTensor):
            in_features = in_features.prod()
        else:
            in_features = int(np.prod(in_features))

        super().__init__(in_features=in_features, out_features=out_features, **kwargs)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        return super().forward(x)


# TODO: This can be only an action layer, no need Linear
class ActionLinear(nn.Module):
    """
    A linear layer that automatically calculates the output shape based on the action_info.

    Parameters
    ----------
    in_features: list
        Size of each input sample
    action_info: dict
        Dict containing information about the environment actions (e.g. shape).
    """

    def __init__(self, in_features, action_info, **kwargs):
        super().__init__()

        self.action_info = action_info
        out_features = int(np.prod(action_info["shape"]))

        self.linear = FlattenLinear(
            in_features=in_features, out_features=out_features, **kwargs
        )

        if action_info.space == "continuous":
            self.log_std = nn.Parameter(torch.zeros(1, out_features))
            # Tiny layer for maximizing exploration
            self.linear.weight.data.normal_(std=0.01)

    def forward(self, x):
        if self.action_info.space == "discrete":
            logits = self.linear(x)
            return logits

        elif self.action_info.space == "continuous":
            mean = self.linear(x)
            log_std = self.log_std.expand_as(mean)
            return torch.stack((mean, log_std), dim=-1)

        else:
            raise ValueError(
                "Action space {} not implemented".format(self.action_info.space)
            )
