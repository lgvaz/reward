import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence

from torchrl.models import BasePGModel


class SurrogatePGModel(BasePGModel):
    def train(self, batch, num_epochs=1):
        batch = batch.apply_to_all(self._to_tensor)

        with torch.no_grad():
            parameters = self.forward(batch.state_t)
            self.memory.old_dists = self.create_dist(parameters)
            batch.log_prob = self.memory.old_dists.log_prob(batch.action).sum(-1)

        for _ in range(num_epochs):
            parameters = self.forward(batch.state_t)
            self.memory.new_dists = self.create_dist(parameters)
            batch.new_log_prob = self.memory.new_dists.log_prob(batch.action).sum(-1)

            loss = self.optimizer_step(batch)
            if self.logger is not None:
                self.logger.add_log('Policy NN Loss', loss.item(), precision=3)

        if self.logger is not None:
            entropy = self.memory.new_dists.entropy().mean()
            self.logger.add_log('Policy Entropy', entropy.item(), precision=3)
        self.memory.clear()

    def add_losses(self, batch):
        self.surrogate_pg_loss(batch)

    def surrogate_pg_loss(self, batch):
        prob_ratio = self.calculate_prob_ratio(batch.new_log_prob, batch.log_prob)
        surrogate = prob_ratio * batch.advantage

        loss = -surrogate.mean()

        self.losses.append(loss)

    def calculate_prob_ratio(self, new_log_probs, old_log_probs):
        prob_ratio = (new_log_probs - old_log_probs).exp()
        return prob_ratio
