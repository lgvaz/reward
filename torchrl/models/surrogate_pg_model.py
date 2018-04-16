import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence

from torchrl.models import BasePGModel


class SurrogatePGModel(BasePGModel):
    def train(self, batch, num_epochs=1):
        batch = batch.apply_to_all(self._to_tensor)

        with torch.no_grad():
            batch.log_prob = self.extract_log_probs(batch.action, self.memory.dists)

        for _ in range(num_epochs):
            parameters = self.forward(batch.state_t)
            self.memory.new_dists = [self.create_dist(p) for p in parameters]
            batch.new_log_prob = self.extract_log_probs(batch.action,
                                                        self.memory.new_dists)

            loss = self.optimizer_step(batch)
            print('Policy loss: {}'.format(loss))

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
