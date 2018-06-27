from abc import ABC, abstractmethod
import torch
from torch.optim import Adam
import torchrl.utils as U


class BaseOpt:
    def __init__(self,
                 model,
                 *,
                 num_epochs=1,
                 num_mini_batches=1,
                 shuffle=True,
                 opt_fn=None,
                 opt_params=None,
                 clip_grad_norm=float('inf'),
                 loss_coef=None):
        self.model = model
        self.num_epochs = num_epochs
        self.num_mini_batches = num_mini_batches
        self.shuffle = shuffle
        self.clip_grad_norm = clip_grad_norm
        self.loss_coef = loss_coef

        opt_fn = opt_fn or Adam
        opt_params = opt_params or dict()
        self.opt = self._create_opt(opt_fn=opt_fn, opt_params=opt_params)

    @property
    @abstractmethod
    def batch_keys(self):
        pass

    @property
    @abstractmethod
    def callbacks(self):
        pass

    @abstractmethod
    def model_parameters(self):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    def _create_opt(self, opt_fn, opt_params):
        return opt_fn(self.model_parameters(), **opt_params)

    def optimizer_step(self, batch):
        '''
        Apply the gradients in respect to the losses defined by :meth:`add_losses`.

        Should use the batch to compute and apply gradients to the network.
        '''
        self.opt.zero_grad()
        loss = self.calculate_loss(batch)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model_parameters(),
                                              self.clip_grad_norm)
        self.opt.step()

        # self.memory.grad_norm.append(norm)
        # self.num_updates += 1

    def learn_from_batch(self, batch, step):
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
        # TODO: Currently always CUDA if possible (no choice)
        batch = batch.apply_to_all(U.to_tensor)

        if self.callbacks.on_train_start(batch):
            return

        for i_epoch in range(self.num_epochs):
            if self.callbacks.on_epoch_start(batch):
                break

            for mini_batch in batch.sample_keys(
                    keys=self.batch_keys,
                    num_mini_batches=self.num_mini_batches,
                    shuffle=self.shuffle):
                if self.callbacks.on_mini_batch_start(mini_batch):
                    break

                self.optimizer_step(mini_batch)

                if self.callbacks.on_mini_batch_end(mini_batch):
                    break

            if self.callbacks.on_epoch_end(batch):
                break

        if self.callbacks.on_train_end(batch):
            return

        if self.callbacks.cleanups(batch):
            return
