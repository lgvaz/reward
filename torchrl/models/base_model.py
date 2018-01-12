from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from torchrl.nn import ModuleExtended
from torchrl.utils import Config


class BaseModel(ModuleExtended, ABC):
    '''
    Basic TorchRL model. The model should take two PyTorch networks (body and head)
    and chain them together.

    Parameters
    ----------
    nn_bodies: Config
        A configuration object containing sections with pytorch networks.
    nn_heads: Config
        A configuration object containing sections with pytorch networks.
    cuda_default: bool
        If True and cuda is supported, use it.
    '''

    def __init__(self, nn_bodies, nn_heads, cuda_default=True):
        super().__init__()

        self.num_updates = 0

        assert isinstance(nn_bodies, Config) and isinstance(nn_heads, Config), \
            'nn_bodies and nn_heads must be of type {}'.format(Config.__name__)
        self.nn_body = nn_bodies
        self.nn_head = nn_heads

        self.cuda_enabled = cuda_default and torch.cuda.is_available()
        if self.cuda_enabled:
            for module in self.nn_body.values():
                module.cuda()
            for module in self.nn_head.values():
                module.cuda()

        # This is needed for pytorch to register this modules as part of this class
        self.nn_body_modules = nn.Sequential(self.nn_body.as_dict())
        self.nn_head_modules = nn.Sequential(self.nn_head.as_dict())

        self.opt = self._create_optimizer()

    def _create_optimizer(self):
        '''
        Creates an optimizer for the model.

        Returns
        -------
        torch.optim
            A pytorch optimizer.

        Examples
        --------
        It's possible to create an optimizer with the same
        configurations for all the model::

            opt = torch.optim.Adam(self.parameters(), lr=1e-2)

        Or use a different configuration for different parts of the model::

            parameters_body = [
                dict(params=module.parameters()) for module in self.nn_body.values()
            ]
            parameters_head = [
                dict(params=module.parameters()) for module in self.nn_head.values()
            ]
            parameters_total = parameters_body + parameters_head

            opt = torch.optim.Adam(parameters_total, lr=1e-2)

        For more information see
        `here <http://pytorch.org/docs/0.3.0/optim.html#per-parameter-options>`_.
        '''
        return torch.optim.Adam(self.parameters(), lr=1e-2)
        # parameters_body = [
        #     dict(params=module.parameters()) for module in self.nn_body.values()
        # ]
        # parameters_head = [
        #     dict(params=module.parameters()) for module in self.nn_head.values()
        # ]
        # parameters_total = parameters_body + parameters_head
        # return torch.optim.Adam(parameters_total, lr=1e-2)

    @abstractmethod
    def forward(self, x):
        '''
        This method should be overwritten by a subclass.

        Should define how the networks are connected.

        Parameters
        ----------
        x: numpy.ndarray
            The environment state.
        '''
        pass
        # return self.nn_head(self.nn_body(x))

    @abstractmethod
    def select_action(self, state):
        '''
        This method should be overwritten by a subclass.

        It should receive the state and select an action based on it.

        Returns
        -------
        action: int or numpy.ndarray
        '''
        pass

    @abstractmethod
    def train(self, batch=None):
        '''
        This method should be inherited by a subclass.

        Should use the batch to compute and apply gradients to the network.

        Parameters
        ----------
        batch: dict
            The batch should contain all the information necessary
            to compute the gradients.
        '''
        self.num_updates += 1

    @classmethod
    def from_config(cls, config, state_shape, action_shape):
        '''
        Creates a model from a configuration file.

        Returns
        -------
        torchrl.models
            A TorchRL model.
        '''
        nn_body = config.model.nn_body.obj.from_config(
            config.model.nn_body.arch.as_dict(),
            kwargs=config.model.nn_body.kwargs.as_dict())
        nn_head = config.model.nn_head.obj(
            input_shape=nn_body.get_output_shape(state_shape),
            output_shape=action_shape,
            **config.model.nn_head.kwargs.as_dict())

        return cls(nn_body, nn_head, **config.model.kwargs.as_dict())
