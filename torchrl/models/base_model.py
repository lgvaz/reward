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
    model: nn.Module
        A pytorch model.
    env: torchrl.envs
            A torchrl environment.
    opt_fn: torch.optim
        The optimizer reference function (the constructor, not the instance)
        (Default is Adam).
    opt_params: dict
        Parameters for the optimizer (Default is empty dict).
    clip_grad_norm: float
        Clip norm for the gradients, if `None` gradients
        will not be clipped (Default is None).
    cuda_default: bool
        If True and cuda is supported, use it (Default is True).
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
        Append losses to ``self.losses``, the losses are used
        at :meth:`optimizer_step` for calculating the gradients.

        Parameters
        ----------
        batch: dict
            The batch should contain all the information necessary
            to compute the gradients.
        '''

    @abstractmethod
    def train(self, batch):
        '''
        The main training loop.

        Parameters
        ----------
        batch: torchrl.utils.Batch
            The batch should contain all the information necessary
            to compute the gradients.
        '''

    def optimizer_step(self, batch):
        '''
        Apply the gradients in respect to the losses defined by :meth:`add_losses`.

        Should use the batch to compute and apply gradients to the network.

        Parameters
        ----------
        batch: torchrl.utils.Batch
            The batch should contain all the information necessary
            to compute the gradients.
        '''
        self.add_losses(batch)

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
        Defines the computation performed at every call.

        Parameters
        ----------
        x: numpy.ndarray
            The environment state.
        '''
        return self.model(x)

    def attach_logger(self, logger):
        '''
        Register a logger to this model.

        Parameters
        ----------
        logger: torchrl.utils.logger
        '''
        self.logger = logger

    def write_logs(self, batch):
        pass

    @classmethod
    def from_config(cls, config, env=None, **kwargs):
        '''
        Creates a model from a configuration file.

        Parameters
        ----------
        config: Config
            Should contatin at least a network definition (``nn_config`` section).
        env: torchrl.envs
            A torchrl environment (Default is None and must be present in the config).
        kwargs: key-word arguments
            Extra arguments that will be passed to the class constructor.

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

    # TODO: Reimplement method
    @classmethod
    def from_file(cls, file_path, *args, **kwargs):
        config = U.Config.load(file_path)

        return cls.from_config(config, *args, **kwargs)
