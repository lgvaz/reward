from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn

from torchrl.nn import ModuleExtended, SequentialExtended
from torchrl.utils import Config, get_obj


class BaseModel(ModuleExtended, ABC):
    '''
    Basic TorchRL model. Takes two :obj:`Config` objects that identify
    the body(ies) and head(s) of the model.

    Parameters
    ----------
    nn_body: Config
        A configuration object containing sections with pytorch networks.
    nn_head: Config
        A configuration object containing sections with pytorch networks.
    cuda_default: bool
        If True and cuda is supported, use it.
    '''

    def __init__(self, config, cuda_default=True):
        super().__init__()

        self.num_updates = 0

        assert isinstance(nn_body, Config) and isinstance(nn_head, Config), \
            'nn_bodies and nn_heads must be of type {}'.format(Config.__name__)
        self.nn_body = config.nn_body
        self.nn_head = config.nn_head

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
        # Prepare for creating body network dict
        body_dict = OrderedDict()
        body_configs = iter(config.nn_body.arch.items())

        # Get first layer
        key, obj_config = next(body_configs)
        assert 'Input' in obj_config['func'].__name__, \
            'The first layer of the network must be an Input layer'
        obj_config['input_shape'] = state_shape
        body_dict[key] = get_obj(obj_config)

        # Get other layers
        for key, obj_config in body_configs:
            body_dict[key] = get_obj(obj_config)
        # Encapsulate all layers
        nn_body = SequentialExtended(body_dict)

        # Prepare for creating head network dict
        head_dict = OrderedDict()
        head_configs = iter(config.nn_head.arch.items())

        # Get first layer
        key, obj_config = next(head_configs)
        assert 'Input' in obj_config['func'].__name__, \
            'The first layer of the network must be an Input layer'
        obj_config['input_shape'] = nn_body.get_output_shape(state_shape)
        head_dict[key] = get_obj(obj_config)

        # TODO: Output is also dynamic!!
        # Get other layers
        for key, obj_dict in head_configs:
            head_dict[key] = get_obj(obj_dict)
        # Encapsulate all layers
        nn_head = SequentialExtended(body_dict)

        return cls(nn_body=nn_body, nn_head=nn_head, **config.model.kwargs.as_dict())

    @classmethod
    def from_file(cls, file_path, *args, **kwargs):
        config = Config.load(file_path)

        return cls.from_config(config, *args, **kwargs)
