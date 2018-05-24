import torch
import torchrl.utils as U
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
        self.kl_target = U.make_callable(kl_target)
        self.kl_penalty = kl_penalty

    def add_losses(self, batch):
        self.surrogate_pg_loss(batch)
        self.kl_penalty_loss(batch)
        self.hinge_loss(batch)

    def kl_penalty_loss(self, batch):
        kl_div = kl_divergence(self.memory.old_dists,
                               self.memory.new_dists).sum(-1).mean()

        loss = self.kl_penalty * kl_div

        self.losses.append(loss)

    def hinge_loss(self, batch):
        kl_div = kl_divergence(self.memory.old_dists,
                               self.memory.new_dists).sum(-1).mean()

        loss = 50 * max(0, kl_div - 2. * self.kl_target(self.step))**2

        self.losses.append(loss)

    def train_step(self, batch):
        with torch.no_grad():
            parameters = self.forward(batch.state_t)
            self.memory.old_dists = self.create_dist(parameters)
            batch.log_prob = self.memory.old_dists.log_prob(batch.action).sum(-1)

        self.add_new_dist(batch)
        for i_iter in range(self.num_epochs):
            self.optimizer_step(batch)

            # Create new policy
            self.add_new_dist(batch)

            if batch.kl_div > 4 * self.kl_target(self.step):
                print('Early stopping')
                break

        # Adjust KL penalty
        if batch.kl_div < self.kl_target(self.step) / 1.5:
            self.kl_penalty /= 2
        if batch.kl_div > self.kl_target(self.step) * 1.5:
            self.kl_penalty *= 2

    def write_logs(self, batch):
        super().write_logs(batch)

        self.logger.add_log(
            self.name + '/KL Target', self.kl_target(self.step), precision=4)
        self.logger.add_log(self.name + '/KL Penalty', self.kl_penalty, precision=4)
