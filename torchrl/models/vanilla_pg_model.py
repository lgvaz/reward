import torch
import torch.nn.functional as F

import torchrl.utils as U
from torchrl.models import BasePGModel


class VanillaPGModel(BasePGModel):
    '''
    The classical Policy Gradient algorithm.
    '''

    def add_losses(self, batch):
        '''
        Define all losses used for calculating the gradient.

        Parameters
        ----------
        batch: Batch
            The batch should contain all the information necessary
            to compute the gradients.
        '''
        self.pg_loss(batch)

    def pg_loss(self, batch):
        '''
        Compute loss based on the policy gradient theorem.

        Parameters
        ----------
        batch: Batch
            The batch should contain all the information necessary
            to compute the gradients.
        '''
        objective = batch.log_prob * batch.advantage
        loss = -objective.mean()

        self.losses.append(loss)

    def train(self, batch):
        batch = batch.apply_to_all(self._to_tensor)
        parameters = self.forward(batch.state_t)
        dists = self.create_dist(parameters)
        batch.log_prob = dists.log_prob(batch.action).sum(-1)

        loss = self.optimizer_step(batch)
        if self.logger is not None:
            self.logger.add_log('Policy NN Loss', loss.item(), precision=3)

        self.memory.clear()
