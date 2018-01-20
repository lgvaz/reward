import torch
import torch.nn.functional as F

import torchrl.utils as U
from torchrl.models import PGModel


class ReinforceModel(PGModel):
    '''
    REINFORCE model.
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
        log_probs = [
            dist.log_prob(action)
            for dist, action in zip(self.saved_dists, batch['actions'])
        ]
        log_probs = torch.cat(log_probs).view(-1)

        objective = log_probs * batch['advantages']
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

    def train(self, batch, logger=None):
        batch['advantages'] = self._to_variable(batch['advantages'])
        batch['actions'] = self._to_variable(batch['actions'].astype('int'))

        super().train(batch=batch, logger=logger)

        self.saved_dists = []
