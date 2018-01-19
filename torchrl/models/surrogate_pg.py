import torch
import torch.nn.functional as F

from torchrl.distributions import CategoricalDist
from torchrl.models import PGModel


class SurrogatePGModel(PGModel):
    def __init__(self, policy_nn_config, value_nn_config=None, share_body=False,
                 **kwargs):

        super().__init__(
            policy_nn_config=policy_nn_config,
            value_nn_config=value_nn_config,
            share_body=share_body,
            **kwargs)

    def train(self, batch, logger=None):
        super().train(batch, logger)

    def forward(self, x):
        '''
        Uses the network to compute action scores
        and apply softmax to obtain action probabilities.

        Parameters
        ----------
        x: numpy.ndarray
            The environment state.

        Returns
        -------
        numpy.ndarray
            Action probabilities
        '''
        action_scores = self.policy_nn.head(self.policy_nn.body(x))
        action_probs = F.softmax(action_scores, dim=1)

        if self.value_nn is not None:
            state_value = self.value_nn.head(self.value_nn.body(x))
            self.saved_state_values.append(state_value)

        return action_probs

    def add_surrogate_pg_loss(self, batch):
        new_probs = F.softmax(self.policy_nn.head(self.policy_nn.body(batch['state_ts'])))
        new_dists = [CategoricalDist(p) for p in new_probs]

        actions = [old_dist.last_action for old_dist in self.saved_dists]
        # TODO: Using this repacking hack to get around diferentiation
        old_logprobs = torch.cat(
            [old_dist.last_log_prob for old_dist in self.saved_dists]).detach()
        new_logprobs = torch.cat(
            [new_dist.log_prob(action) for new_dist, action in zip(new_dists, actions)])

        prob_ratio = (new_logprobs - old_logprobs).exp()

        advantages = self._to_variable(batch['advantages'])
        surrogate_obj = prob_ratio * advantages

        loss = -surrogate_obj.sum()
        self.losses.append(loss)

    def add_value_nn_loss(self, vtarget):
        state_values = torch.cat(self.saved_state_values).view(-1)

        loss = F.mse_loss(input=state_values, target=vtarget)

        self.losses.append(loss)

    def add_losses(self, batch):
        self.add_surrogate_pg_loss(batch)

        vtarget = self._to_variable(batch['vtarget'])
        self.add_value_nn_loss(vtarget)
