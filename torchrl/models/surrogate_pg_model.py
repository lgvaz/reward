import torch
import torch.nn.functional as F

from torchrl.distributions import CategoricalDist
from torchrl.models import PGModel


class SurrogatePGModel(PGModel):
    # def __init__(self, policy_nn_config, value_nn_config=None, share_body=False,
    #              **kwargs):
    #     super().__init__(
    #         policy_nn_config=policy_nn_config,
    #         value_nn_config=value_nn_config,
    #         share_body=share_body,
    #         **kwargs)

    def train(self, batch, num_epochs=1, logger=None):
        batch['actions'] = self._to_variable(batch['actions'].astype('int'))
        batch['advantages'] = self._to_variable(batch['advantages'])
        # .detach() is used so no gradients are computed w.r.t. old_log_probs
        batch['old_log_probs'] = torch.cat([
            dist.log_prob(action)
            for dist, action in zip(self.saved_dists, batch['actions'])
        ]).detach()

        for _ in range(num_epochs):
            super().train(batch, logger)

        self.saved_dists = []

    def calculate_prob_ratio(self, batch):
        new_probs = F.softmax(self.policy_nn.head(self.policy_nn.body(batch['state_ts'])))
        new_dists = [CategoricalDist(p) for p in new_probs]

        new_log_probs = torch.cat([
            new_dist.log_prob(action)
            for new_dist, action in zip(new_dists, batch['actions'])
        ])

        prob_ratio = (new_log_probs - batch['old_log_probs']).exp()

        return prob_ratio

    def add_surrogate_pg_loss(self, batch):
        prob_ratio = self.calculate_prob_ratio(batch)
        surrogate = prob_ratio * batch['advantages']

        loss = -surrogate.sum()
        self.losses.append(loss)

    def add_losses(self, batch):
        self.add_surrogate_pg_loss(batch)
        self.add_value_nn_loss(batch)
