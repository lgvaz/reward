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
        print('Policy loss: {}'.format(loss))

        self.losses.append(loss)

    def train(self, batch):
        batch = batch.apply_to_all(self._to_tensor)
        batch.log_prob = self.extract_log_probs(batch.action, self.memory.dists)

        self.optimizer_step(batch)

        self.memory.clear()
