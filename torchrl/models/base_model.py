from abc import ABC, abstractmethod, abstractproperty
from collections import ChainMap

import os
import numpy as np
import torch
import torch.nn as nn

import torchrl.utils as U
from torchrl.nn import ModuleExtended

from multiprocessing import Process


def profile(x):
    return lambda *args, **kwargs: x(*args, **kwargs)


class BaseModel(ModuleExtended, ABC):
    '''
    Basic TorchRL model. Takes two :obj:`Config` objects that identify
    the body(ies) and head(s) of the model.

    Parameters
    ----------
    model: nn.Module
        A pytorch model.
    batcher: torchrl.batcher
        A torchrl batcher.
    num_epochs: int
        How many times to train over the entire dataset (Default is 1).
    num_mini_batches: int
        How many mini-batches to subset the batch
        (Default is 1, so all the batch is used at once).
    opt_fn: torch.optim
        The optimizer reference function (the constructor, not the instance)
        (Default is Adam).
    opt_params: dict
        Parameters for the optimizer (Default is empty dict).
    clip_grad_norm: float
        Max norm of the gradients, if float('inf') no clipping is done
        (Default is float('inf')).
    loss_coef: float
        Used when sharing networks, should balance the contribution
        of the grads of each model.
    cuda_default: bool
        If True and cuda is supported, use it (Default is True).
    '''

    def __init__(self,
                 model,
                 batcher,
                 *,
                 num_epochs=1,
                 num_mini_batches=1,
                 opt_fn=None,
                 opt_params=None,
                 lr_schedule=None,
                 clip_grad_norm=float('inf'),
                 loss_coef=1.,
                 cuda_default=True):
        super().__init__()

        self.model = model
        self.batcher = batcher
        self.num_epochs = num_epochs
        self.num_mini_batches = num_mini_batches
        self.lr_schedule = U.make_callable(lr_schedule or opt_params['lr'])
        self.clip_grad_norm = clip_grad_norm
        self.loss_coef_fn = U.make_callable(loss_coef)

        self.memory = U.DefaultMemory()
        self.num_updates = 0
        self.step = 0
        self.losses = []
        self.register_losses()
        self.callbacks = U.Callback()
        self.register_callbacks()
        self.logger = None

        # Enable cuda if wanted
        self.cuda_enabled = cuda_default and torch.cuda.is_available()
        if self.cuda_enabled:
            self.model.cuda()
        # self.device = torch.device('cuda' if cuda_default and torch.cuda.is_available()
        #                            else 'cpu')
        # self.to(self.device)

        # Create optimizer
        opt_params = opt_params or dict()
        opt_fn = opt_fn or torch.optim.Adam
        self.opt = opt_fn(self.parameters(), **opt_params)

    @abstractproperty
    def batch_keys(self):
        '''
        The batch keys needed for computing all losses.
        This is done to reduce overhead when sampling a dataloader,
        it makes sure only the requested keys are being sampled.
        '''

    @abstractmethod
    def register_losses(self):
        '''
        Append losses to ``self.losses``, the losses are used
        at :meth:`optimizer_step` for calculating the gradients.

        Parameters
        ----------
        batch: dict
            The batch should contain all the information necessary
            to compute the gradients.
        '''

    @staticmethod
    @abstractmethod
    def output_layer(input_shape, action_info):
        '''
        The final layer of the model, will be appended to the model head.

        Parameters
        ----------
        input_shape: int or tuple
            The shape of the input to this layer.
        action_info: dict
            Dictionary containing information about the action space.

        Examples
        --------
        The output of most PG models have the same dimension as the action,
        but the output of the Value models is rank 1. This is where this is defined.
        '''

    @property
    def body(self):
        return self.model.layers[0]

    @property
    def head(self):
        return self.model.layers[1]

    @property
    def lr(self):
        return self.lr_schedule(self.step)

    @property
    def loss_coef(self):
        return self.loss_coef_fn(self.step)

    @property
    def name(self):
        return self.__class__.__name__

    def register_loss(self, func):
        self.losses.append(func)

    def register_callbacks(self):
        pass

    def calculate_loss(self, batch):
        losses = {f.__name__: f(batch) for f in self.losses}
        self.memory.losses.append(losses)

        return sum(losses.values())

    def learn_from_batch(self, batch, num_epochs, num_mini_batches, shuffle=True):
        '''
        Define the model training procedure.

        Parameters
        ----------
        batch: torchrl.utils.Batch
            The batch should contain all the information necessary
            to compute the gradients.
        num_epochs: int
            How many times to train over the entire dataset.
        num_mini_batches: int
            How many mini-batches to subset the batch.
        shuffle: bool
            Whether to shuffle dataset.
        '''
        for i_epoch in range(num_epochs):
            if self.callbacks.on_epoch_start(batch):
                break

            for mini_batch in batch.sample_keys(
                    keys=self.batch_keys, num_mini_batches=num_mini_batches,
                    shuffle=shuffle):
                if self.callbacks.on_mini_batch_start(mini_batch):
                    break

                self.optimizer_step(mini_batch)

                if self.callbacks.on_mini_batch_end(mini_batch):
                    break

            if self.callbacks.on_epoch_end(batch):
                break

    def train_step(self, batch):
        '''
        The most basic learning step. Just calls :func:`self.learn_from_batch`
        '''
        if self.callbacks.on_train_start(batch):
            return

        self.learn_from_batch(
            batch, num_epochs=self.num_epochs, num_mini_batches=self.num_mini_batches)

        if self.callbacks.on_train_end(batch):
            return

    @profile
    def train(self, batch, step):
        '''
        Wrapper around :meth:`train_step`, adds functionalities
        to before and after the training step.

        Parameters
        ----------
        batch: torchrl.utils.Batch
            The batch should contain all the information necessary
            to compute the gradients.
        '''
        self.step = step
        self.set_lr(value=self.lr)
        # TODO: Transform to tensor only one time, maybe in agent
        # batch = batch.apply_to_all(self.to_tensor)
        # batch = batch.apply_to_keys(func=self.to_tensor, keys=self.batch_keys)

        self.train_step(batch)

        if self.logger is not None:
            self.write_logs(batch)

        self.memory.clear()

    def optimizer_step(self, batch):
        '''
        Apply the gradients in respect to the losses defined by :meth:`add_losses`.

        Should use the batch to compute and apply gradients to the network.
        '''
        self.opt.zero_grad()
        loss = self.calculate_loss(batch) * self.loss_coef
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
        self.opt.step()

        self.memory.grad_norm.append(norm)
        self.num_updates += 1

    def forward(self, x):
        '''
        Defines the computation performed at every call.

        Parameters
        ----------
        x: numpy.ndarray
            The environment state.
        '''
        return self.model(x)

    def set_lr(self, value):
        '''
        Change the learning rate of the optimizer.

        Parameters
        ----------
        value: float
            The new learning rate.
        '''
        for param_group in self.opt.param_groups:
            param_group['lr'] = value

    def attach_logger(self, logger):
        '''
        Register a logger to this model.

        Parameters
        ----------
        logger: torchrl.utils.logger
        '''
        self.logger = logger

    def wrap_name(self, name):
        return '/'.join([self.name, name])

    def add_log(self, name, value, **kwargs):
        self.logger.add_log(name=self.wrap_name(name), value=value, **kwargs)

    def add_tf_only_log(self, name, value, **kwargs):
        self.logger.add_tf_only_log(name=self.wrap_name(name), value=value, **kwargs)

    def add_debug_log(self, name, value, **kwargs):
        self.logger.add_debug(name=self.wrap_name(name), value=value, **kwargs)

    def add_histogram_log(self, name, values, **kwargs):
        self.logger.add_histogram(name=self.wrap_name(name), values=values, **kwargs)

    def write_logs(self, batch):
        '''
        Write logs to the terminal and to a tf log file.

        Parameters
        ----------
        batch: Batch
            Some logs might need the batch for calculation.
        '''
        self.add_log('LR', self.lr, precision=4)
        self.add_tf_only_log('Grad Norm', np.mean(self.memory.grad_norm))

        total_loss = 0
        # TODO: POSSIBLY WRONG
        for k, v in ChainMap(*self.memory.losses).items():
            loss = v.mean()
            total_loss += loss
            self.add_tf_only_log('/'.join(['Loss', k]), loss, precision=4)

        self.add_log('Loss/Total', total_loss, precision=4)

    # TODO: Env and batcher are needed?
    @classmethod
    def from_config(cls, config, batcher=None, body=None, head=None, **kwargs):
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
        # env = env or U.env_from_config(config)
        # config.pop('env', None)

        if not 'body' in config.nn_config:
            config.nn_config.body = []
        if not 'head' in config.nn_config:
            config.nn_config.head = []

        nn_config = config.pop('nn_config')
        model = U.nn_from_config(
            config=nn_config,
            state_info=batcher.get_state_info(),
            action_info=batcher.get_action_info(),
            body=body,
            head=head)

        output_layer = cls.output_layer(
            input_shape=model.get_output_shape(batcher.get_state_info().shape),
            action_info=batcher.get_action_info())

        model.layers.head.append(output_layer)

        return cls(model=model, batcher=batcher, **config.as_dict(), **kwargs)

    @classmethod
    def from_file(cls, file_path, *args, **kwargs):
        config = U.Config.load(file_path)

        return cls.from_config(config, *args, **kwargs)

    @classmethod
    def from_arch(cls, arch, *args, **kwargs):
        module_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(module_path, 'archs', arch)

        return cls.from_file(file_path=path, *args, **kwargs)
