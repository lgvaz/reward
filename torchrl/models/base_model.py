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

    def __init__(self,
                 state_info,
                 action_info,
                 learning_rate,
                 log_dir=None,
                 cuda_default=True):
        super().__init__()

        self.state_info = state_info
        self.action_info = action_info
        self.lr = learning_rate
        self.logger = U.Logger(log_dir=log_dir)
        self.num_updates = 0
        self.networks = []
        self.losses = []

        self.create_networks()

        # Enable cuda if wanted
        self.cuda_enabled = cuda_default and torch.cuda.is_available()
        if self.cuda_enabled:
            for network in self.networks:
                network.cuda()

        # # This is needed for pytorch to register this modules as part of this class
        self.nn_modules = nn.ModuleList(self.networks)

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
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        # parameters_body = [
        #     dict(params=module.parameters()) for module in self.nn_body.values()
        # ]
        # parameters_head = [
        #     dict(params=module.parameters()) for module in self.nn_head.values()
        # ]
        # parameters_total = parameters_body + parameters_head
        # return torch.optim.Adam(parameters_total, lr=1e-2)

    @abstractmethod
    def create_networks(self):
        pass

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
        for _ in range(num_epochs):
            for data in U.DataGenerator(batch, batch_size=64):
                self.optimizer_step(data)

        self.write_logs(batch)

    def optimizer_step(self, batch):
        '''
        Apply the gradients in respect to the losses defined by :func:`add_losses`_.

        Should use the batch to compute and apply gradients to the network.

        Parameters
        ----------
        batch: dict
            The batch should contain all the information necessary
            to compute the gradients.
        '''
        self.add_losses(batch)

        self.opt.zero_grad()
        loss = sum(self.losses)
        loss.backward()
        self.opt.step()

        self.losses = []
        self.num_updates += 1

    def write_logs(self, batch):
        pass

    def net_from_config(self, net_config, body=None, head=None):
        nets = U.nn_from_config(
            config=net_config,
            state_info=self.state_info,
            action_info=self.action_info,
            body=body,
            head=head)

        for net in nets.values():
            self.networks.append(net)

        return nets

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
