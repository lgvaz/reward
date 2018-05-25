import torch
import torch.nn.functional as F

import torchrl.utils as U
from torchrl.models import BasePGModel


class VanillaPGModel(BasePGModel):
    '''
    The classical Policy Gradient algorithm.
    '''

    def add_losses(self, batch):
        self.pg_loss(batch)
        self.entropy_loss(batch)

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
        parameters = self.forward(batch.state_t)
        dists = self.create_dist(parameters)
        batch.log_prob = dists.log_prob(batch.action).sum(-1)
        self.memory.entropy = dists.entropy().mean()

        loss = self.optimizer_step(batch)

    def write_logs(self, batch):
        super().write_logs(batch)

        entropy = self.memory.new_dists.entropy().mean()
        self.logger.add_log(self.name + '/Entropy', entropy)
