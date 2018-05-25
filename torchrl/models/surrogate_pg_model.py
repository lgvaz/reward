import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence

from torchrl.models import BasePGModel


class SurrogatePGModel(BasePGModel):
    r'''
    The Surrogate Policy Gradient algorithm instead maximizes a "surrogate" objective, given by:

    .. math::

        L^{CPI}({\theta}) = \hat{E}_t \left[\frac{\pi_{\theta}(a|s)}
        {\pi_{\theta_{old}}(a|s)} \hat{A} \right ]

    Parameters
    ----------
    num_epochs: int
        How many times to train over the entire dataset (Default is 10).
    '''

    def __init__(self, model, env, num_epochs=1, mini_batch_size=512, **kwargs):
        super().__init__(model=model, env=env, **kwargs)
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size

    @property
    def kl_div(self):
        return kl_divergence(self.memory.old_dists, self.memory.new_dists).sum(-1).mean()

    @property
    def entropy(self):
        return self.memory.new_dists.entropy().mean()

    def add_new_dist(self, batch):
        parameters = self.forward(batch.state_t)
        self.memory.new_dists = self.create_dist(parameters)
        # batch.new_log_prob = self.memory.new_dists.log_prob(batch.action).sum(-1)

    def train_step(self, batch):
        with torch.no_grad():
            parameters = self.forward(batch.state_t)
            self.memory.old_dists = self.create_dist(parameters)
            batch.log_prob = self.memory.old_dists.log_prob(batch.action).sum(-1)

        self.memory.batch_keys.extend(['state_t', 'action', 'log_prob', 'advantage'])
        for i_iter in range(self.num_epochs):
            for mini_batch in batch.sample_keys(
                    keys=self.memory.batch_keys,
                    batch_size=self.mini_batch_size,
                    shuffle=True):
                self.add_new_dist(mini_batch)
                self.optimizer_step(mini_batch)

            # self.optimizer_step(batch)
            # self.add_new_dist(batch)

        # Create new policy on complete batch
        self.add_new_dist(batch)

    def write_logs(self, batch):
        super().write_logs(batch)

        self.logger.add_log(self.name + '/Entropy', self.entropy)
        self.logger.add_log(self.name + '/KL Divergence', self.kl_div, precision=4)

    def add_losses(self, batch):
        self.surrogate_pg_loss(batch)
        self.entropy_loss(batch)

    def surrogate_pg_loss(self, batch):
        '''
        The surrogate pg loss, as described before.

        Parameters
        ----------
            batch: Batch
        '''
        prob_ratio = self.calculate_prob_ratio(batch.new_log_prob, batch.log_prob)
        surrogate = prob_ratio * batch.advantage

        loss = -surrogate.mean()

        self.losses.append(loss)

    def calculate_prob_ratio(self, new_log_probs, old_log_probs):
        '''
        Calculates the probability ratio between two policies.

        Parameters
        ----------
        new_log_probs: torch.Tensor
        old_log_probs: torch.Tensor
        '''
        prob_ratio = (new_log_probs - old_log_probs).exp()
        return prob_ratio
