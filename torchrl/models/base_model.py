from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import torchrl.utils as U
from torchrl.nn import ModuleExtended


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

    def __init__(self, model, opt=None, logger=None, cuda_default=True):
        super().__init__()

        self.model = model
        self.logger = logger

        self.memory = U.SimpleMemory()
        self.num_updates = 0
        self.losses = []

        # Enable cuda if wanted
        self.cuda_enabled = cuda_default and torch.cuda.is_available()
        if self.cuda_enabled:
            self.model.cuda()

        # TODO: Rework opt design
        self.opt = opt or torch.optim.Adam(self.parameters(), lr=3e-4)

    # def _create_optimizer(self, lr=1e-3, **kwargs):
    #     '''
    #     Creates an optimizer for the model.

    #     Returns
    #     -------
    #     torch.optim
    #         A pytorch optimizer.

    #     Examples
    #     --------
    #     It's possible to create an optimizer with the same
    #     configurations for all the model::

    #         opt = torch.optim.Adam(self.parameters(), lr=1e-2)

    #     Or use a different configuration for different parts of the model::

    #         parameters_body = [
    #             dict(params=module.parameters()) for module in self.nn_body.values()
    #         ]
    #         parameters_head = [
    #             dict(params=module.parameters()) for module in self.nn_head.values()
    #         ]
    #         parameters_total = parameters_body + parameters_head

    #         opt = torch.optim.Adam(parameters_total, lr=1e-2)

    #     For more information see
    #     `here <http://pytorch.org/docs/0.3.0/optim.html#per-parameter-options>`_.
    #     '''
    #     return torch.optim.Adam(self.parameters(), lr=lr, **kwargs)
    # parameters_body = [
    #     dict(params=module.parameters()) for module in self.nn_body.values()
    # ]
    # parameters_head = [
    #     dict(params=module.parameters()) for module in self.nn_head.values()
    # ]
    # parameters_total = parameters_body + parameters_head
    # return torch.optim.Adam(parameters_total, lr=1e-2)

    def forward(self, x):
        '''
        This method should be overwritten by a subclass.

        Should define how the networks are connected.

        Parameters
        ----------
        x: numpy.ndarray
            The environment state.
        '''
        return self.model(x)

    # @abstractmethod
    # def select_action(self, state):
    #     '''
    #     This method should be overwritten by a subclass.

    #     It should receive the state and select an action based on it.

    #     Returns
    #     -------
    #     action: int or numpy.ndarray
    #     '''

    @abstractmethod
    def add_losses(self, batch):
        '''
        This method should be overwritten by a subclass.

        It should append all the necessary losses to `self.losses`.

        Parameters
        ----------
        batch: dict
            The batch should contain all the information necessary
            to compute the gradients.
        '''

    @abstractmethod
    def train(self, batch, num_epochs=1):
        '''
        Basic train function.

        Perform a optimizer step and write logs

        Parameters
        ----------
        batch: dict
            The batch should contain all the information necessary
            to compute the gradients.
        num_epochs: int
            Number of times to fit on the same batch.
        '''
        pass

    def optimizer_step(self, *args, **kwargs):
        '''
        Apply the gradients in respect to the losses defined by :func:`add_losses`_.

        Should use the batch to compute and apply gradients to the network.

        Parameters
        ----------
        batch: dict
            The batch should contain all the information necessary
            to compute the gradients.
        '''
        self.add_losses(*args, **kwargs)

        self.opt.zero_grad()
        loss = sum(self.losses)
        loss.backward()
        self.opt.step()

        self.losses = []
        self.num_updates += 1

        return loss

    def write_logs(self, batch):
        pass

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        '''
        Creates a model from a configuration file.

        Returns
        -------
        torchrl.models
            A TorchRL model.
        '''
        return cls(*args, **config.as_dict(), **kwargs)

    @classmethod
    def from_file(cls, file_path, *args, **kwargs):
        config = U.Config.load(file_path)

        return cls.from_config(config, *args, **kwargs)
