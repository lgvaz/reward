import torch
from torchrl.models import SurrogatePGModel
from torch.distributions.kl import kl_divergence


class PPOAdaptiveModel(SurrogatePGModel):
    '''
    Proximal Policy Optimization as described in https://arxiv.org/pdf/1707.06347.pdf.


    Parameters
    ----------
    num_epochs: int
        How many times to train over the entire dataset (Default is 10).
    '''

    def __init__(self, model, env, kl_target=0.01, kl_penalty=1., num_epochs=10,
                 **kwargs):
        super().__init__(model=model, env=env, num_epochs=num_epochs, **kwargs)
        self.kl_target = kl_target
        self.kl_penalty = kl_penalty

    def add_losses(self, batch):
        self.surrogate_pg_loss(batch)
        self.kl_penalty_loss(batch)

    def kl_penalty_loss(self, batch):
        kl_div = kl_divergence(self.memory.old_dists,
                               self.memory.new_dists).sum(-1).mean()

        loss = self.kl_penalty * kl_div

        self.losses.append(loss)

    def train(self, batch):
        batch = batch.apply_to_all(self._to_tensor)

        with torch.no_grad():
            parameters = self.forward(batch.state_t)
            self.memory.old_dists = self.create_dist(parameters)
            batch.log_prob = self.memory.old_dists.log_prob(batch.action).sum(-1)

        for _ in range(self.num_epochs):
            parameters = self.forward(batch.state_t)
            self.memory.new_dists = self.create_dist(parameters)
            batch.new_log_prob = self.memory.new_dists.log_prob(batch.action).sum(-1)

            loss = self.optimizer_step(batch)
            if self.logger is not None:
                self.logger.add_log('Policy/Loss', loss.item(), precision=3)

        # Calculate KL div, change penalty when needed
        kl_div = kl_divergence(self.memory.old_dists,
                               self.memory.new_dists).sum(-1).mean()
        if kl_div < self.kl_target / 1.5:
            self.kl_penalty /= 2
        if kl_div > self.kl_target * 1.5:
            self.kl_penalty *= 2

        if self.logger is not None:
            entropy = self.memory.new_dists.entropy().mean()
            self.logger.add_log('Policy/KL Penalty', self.kl_penalty, precision=4)
            self.logger.add_log('Policy/Entropy', entropy.item())
            self.logger.add_log('Policy/KL Divergence', kl_div.item(), precision=4)

        self.memory.clear()
