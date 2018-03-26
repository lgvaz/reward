import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence

from torchrl.models import PGModel


class SurrogatePGModel(PGModel):
    def train(self, batch, num_epochs=1):
        batch['actions'] = self._to_variable(batch['actions'])
        batch['advantages'] = self._to_variable(batch['advantages']).view(-1, 1)
        with torch.no_grad():
            batch['old_log_probs'] = torch.stack([
                dist.log_prob(action).sum()
                for dist, action in zip(self.saved_dists, batch['actions'])
            ])

        super().train(batch=batch, num_epochs=num_epochs)

        self.saved_dists = []

    def calculate_prob_ratio(self, batch, new_dists):
        new_log_probs = torch.stack([
            new_dist.log_prob(action).sum()
            for new_dist, action in zip(new_dists, batch['actions'])
        ])

        prob_ratio = (new_log_probs - batch['old_log_probs']).exp()

        return prob_ratio

    def add_surrogate_pg_loss(self, batch, new_dists):
        prob_ratio = self.calculate_prob_ratio(batch, new_dists)
        surrogate = prob_ratio * batch['advantages']

        loss = -surrogate.sum()
        self.losses.append(loss)

    def add_losses(self, batch):
        new_parameters = self.forward(batch['state_ts'])
        new_dists = [self.create_dist(p) for p in new_parameters]

        self.add_surrogate_pg_loss(batch, new_dists)
        self.add_value_nn_loss(batch)

    def kl_divergence(self, new_dists):
        kl_divs = [
            kl_divergence(old_dist, new_dist).sum()
            for old_dist, new_dist in zip(self.saved_dists, new_dists)
        ]

        return torch.stack(kl_divs).mean()

    def write_logs(self, batch):
        new_parameters = self.forward(batch['state_ts'])
        new_dists = [self.create_dist(p) for p in new_parameters]

        self.logger.add_log('Policy/Entropy', self.entropy(new_dists).item())
        self.logger.add_log(
            'Policy/KL_div', self.kl_divergence(new_dists).item(), precision=5)
