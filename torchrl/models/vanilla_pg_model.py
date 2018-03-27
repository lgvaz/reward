import torch
import torch.nn.functional as F

import torchrl.utils as U
from torchrl.models import BasePGModel


class VanillaPGModel(BasePGModel):
    '''
    The classical Policy Gradient algorithm.
    '''

    def add_pg_loss(self, batch):
        '''
        Compute loss based on the policy gradient theorem.

        Parameters
        ----------
        batch: dict
            The batch should contain all the information necessary
            to compute the gradients.
        '''
        objective = batch['log_probs'] * batch['advantages']
        loss = -objective.sum()

        self.losses.append(loss)

    def add_losses(self, batch):
        '''
        Define all losses used for calculating the gradient.

        Parameters
        ----------
        batch: dict
            The batch should contain all the information necessary
            to compute the gradients.
        '''
        self.add_pg_loss(batch)
        if self.value_nn is not None:
            self.add_value_nn_loss(batch)

    def train(self, batch, num_epochs=1):
        batch['advantages'] = self._to_tensor(batch['advantages'])
        batch['actions'] = self._to_tensor(batch['actions'].astype('int'))

        batch['log_probs'] = torch.stack([
            dist.log_prob(action).sum()
            for dist, action in zip(self.saved_dists, batch['actions'])
        ])

        super().train(batch=batch)

        self.saved_dists = []
