import torch
import torch.nn.functional as F

from torchrl.distributions import CategoricalDist
from torchrl.models import BaseModel


class PGModel(BaseModel):
    def __init__(self, policy_nn_config, value_nn_config=None, share_body=False,
                 **kwargs):
        self.policy_nn_config = policy_nn_config
        self.value_nn_config = value_nn_config
        self.share_body = share_body
        self.saved_dists = []
        self.saved_state_values = []

        super().__init__(**kwargs)

    def create_networks(self):
        self.policy_nn = self.net_from_config(self.policy_nn_config)

        if self.value_nn_config is not None:
            body = self.policy_nn.body if self.share_body else None
            self.value_nn = self.net_from_config(self.value_nn_config, body=body)
        else:
            self.value_nn = None

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

    def select_action(self, state):
        '''
        Uses the values given by ``self.forward`` to select an action.

        If the action space is discrete the values will be assumed to represent
        probabilities of a categorical distribution.

        If the action space is continuous the values will be assumed to represent
        the mean and variance of a normal distribution.

        Parameters
        ----------
        state: numpy.ndarray
            The environment state.

        Returns
        -------
        action: int or numpy.ndarray
        '''
        # TODO: Continuous distribution
        probs = self.forward(state)
        dist = CategoricalDist(probs)
        action = dist.sample()
        self.saved_dists.append(dist)

        return action.data[0]

    def train(self, batch, logger=None):
        super().train(batch=batch, logger=logger)

        self.saved_dists = []
        self.saved_state_values = []

    def add_state_values(self, traj):
        if self.value_nn is not None:
            steps = len(traj['rewards'])

            state_values = torch.cat(self.saved_state_values[-steps:])
            state_values = state_values.data.view(-1).cpu().numpy()
            traj['state_values'] = state_values

        else:
            pass
