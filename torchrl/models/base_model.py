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
                 model,
                 env,
                 opt_fn=None,
                 opt_params=dict(),
                 clip_grad_norm=None,
                 cuda_default=True):
        super().__init__()

        self.model = model
        self.env = env
        self.clip_grad_norm = clip_grad_norm

        self.memory = U.SimpleMemory()
        self.num_updates = 0
        self.losses = []
        self.logger = None

        # Create optimizer
        opt_fn = opt_fn or torch.optim.Adam
        self.opt = opt_fn(self.parameters(), **opt_params)

        # Enable cuda if wanted
        self.cuda_enabled = cuda_default and torch.cuda.is_available()
        if self.cuda_enabled:
            self.model.cuda()

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
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm(self.parameters(), self.clip_grad_norm)
        self.opt.step()

        self.losses = []
        self.num_updates += 1

        return loss

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

    def attach_logger(self, logger):
        self.logger = logger

    def write_logs(self, batch):
        pass

    @classmethod
    def from_config(cls, config, env=None, **kwargs):
        '''
        Creates a model from a configuration file.

        Returns
        -------
        torchrl.models
            A TorchRL model.
        '''
        env = env or U.env_from_config(config)
        config.pop('env', None)

        nn_config = config.pop('nn_config')
        model = U.nn_from_config(nn_config, env.state_info, env.action_info)

        return cls(model=model, env=env, **config.as_dict(), **kwargs)

    @classmethod
    def from_file(cls, file_path, *args, **kwargs):
        config = U.Config.load(file_path)

        return cls.from_config(config, *args, **kwargs)
