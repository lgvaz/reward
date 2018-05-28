from abc import ABC, abstractmethod, abstractproperty

import numpy as np
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
        Clip norm for the gradients, if `None` gradients
        will not be clipped (Default is None).
    loss_coef: float
        Used when sharing networks, should balance the contribution
        of the grads of each model.
    cuda_default: bool
        If True and cuda is supported, use it (Default is True).
    '''

    def __init__(self,
                 model,
                 env,
                 *,
                 num_epochs=1,
                 num_mini_batches=1,
                 opt_fn=None,
                 opt_params=dict(),
                 lr_schedule=None,
                 clip_grad_norm=None,
                 loss_coef=1.,
                 cuda_default=True):
        super().__init__()

        self.model = model
        self.env = env
        self.num_epochs = num_epochs
        self.num_mini_batches = num_mini_batches
        self.lr_schedule = U.make_callable(lr_schedule or opt_params['lr'])
        self.clip_grad_norm = clip_grad_norm
        self.loss_coef = U.make_callable(loss_coef)

        self.memory = U.DefaultMemory()
        self.num_updates = 0
        self.step = 0
        self.losses = []
        self.logger = None
        self.callbacks = U.Callback()
        self.register_callbacks()

        # Create optimizer
        opt_fn = opt_fn or torch.optim.Adam
        self.opt = opt_fn(self.parameters(), **opt_params)

        # Enable cuda if wanted
        self.cuda_enabled = cuda_default and torch.cuda.is_available()
        if self.cuda_enabled:
            self.model.cuda()

    @abstractproperty
    def batch_keys(self):
        '''
        The batch keys needed for computing all losses.
        This is done to reduce overhead when sampling a dataloader,
        it makes sure only the requested keys are being sampled.
        '''

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

    @property
    def body(self):
        return self.model.layers[0]

    @property
    def head(self):
        return self.model.layers[1]

    @property
    def lr(self):
        return self.opt.param_groups[0]['lr']

    @property
    def name(self):
        return self.__class__.__name__

    def register_callbacks(self):
        pass

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

    def train(self, batch):
        '''
        Wrapper around :meth:`train_step`, adds functionalities
        to before and after the training step.

        Parameters
        ----------
        batch: torchrl.utils.Batch
            The batch should contain all the information necessary
            to compute the gradients.
        '''
        self.step = batch.step[-1]
        self.set_lr(value=self.lr_schedule(self.step))
        batch = batch.apply_to_all(self._to_tensor)

        self.train_step(batch)

        if self.logger is not None:
            self.write_logs(batch)

        self.memory.clear()

    def optimizer_step(self, *args, **kwargs):
        '''
        Apply the gradients in respect to the losses defined by :meth:`add_losses`.

        Should use the batch to compute and apply gradients to the network.
        '''
        self.add_losses(*args, **kwargs)

        self.opt.zero_grad()
        loss = sum(self.losses) * self.loss_coef(self.step)
        loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
        self.opt.step()

        self.memory.loss.append(U.to_np(loss))
        self.losses = []
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
        self.add_log('Loss', np.mean(self.memory.loss))
        self.add_log('LR', self.lr_schedule(self.step), precision=4)

    @classmethod
    def from_config(cls, config, env=None, body=None, head=None, **kwargs):
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
        model = U.nn_from_config(
            config=nn_config,
            state_info=env.state_info,
            action_info=env.action_info,
            body=body,
            head=head)

        return cls(model=model, env=env, **config.as_dict(), **kwargs)

    # TODO: Reimplement method
    @classmethod
    def from_file(cls, file_path, *args, **kwargs):
        config = U.Config.load(file_path)

        return cls.from_config(config, *args, **kwargs)
