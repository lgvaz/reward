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

    def __init__(self, model, env, num_epochs=1, **kwargs):
        super().__init__(model=model, env=env, **kwargs)
        self.num_epochs = num_epochs

    def add_new_dist(self, batch):
        parameters = self.forward(batch.state_t)
        self.memory.new_dists = self.create_dist(parameters)
        batch.new_log_prob = self.memory.new_dists.log_prob(batch.action).sum(-1)
        batch.kl_div = kl_divergence(self.memory.old_dists,
                                     self.memory.new_dists).sum(-1).mean()

    def train(self, batch):
        batch = batch.apply_to_all(self._to_tensor)

        with torch.no_grad():
            parameters = self.forward(batch.state_t)
            self.memory.old_dists = self.create_dist(parameters)
            batch.log_prob = self.memory.old_dists.log_prob(batch.action).sum(-1)

        self.add_new_dist(batch)
        for i_iter in range(self.num_epochs):
            loss = self.optimizer_step(batch)

            # Create new policy
            self.add_new_dist(batch)

            if self.logger is not None:
                self.logger.add_log('Policy/Loss', loss.item(), precision=3)

        if self.logger is not None:
            entropy = self.memory.new_dists.entropy().mean()
            self.logger.add_log('Policy/Entropy', entropy.item())
            self.logger.add_log('Policy/KL Divergence', batch.kl_div.item(), precision=4)

        self.memory.clear()

    def add_losses(self, batch):
        self.surrogate_pg_loss(batch)

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
