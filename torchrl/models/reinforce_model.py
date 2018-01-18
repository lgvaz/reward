import torch
import torch.nn.functional as F

import torchrl.utils as U
from torchrl.models import PGModel


class ReinforceModel(PGModel):
    '''
    REINFORCE model.
    '''

    def __init__(self,
                 policy_nn_config,
                 value_nn_config=None,
                 share_body=False,
                 normalize_advantages=True,
                 **kwargs):
        self.policy_nn_config = policy_nn_config
        self.value_nn_config = value_nn_config
        self.share_body = share_body
        self.normalize_advantages = normalize_advantages
        self.saved_log_probs = []
        self.saved_state_values = []

        super().__init__(
            policy_nn_config=policy_nn_config, value_nn_config=value_nn_config, **kwargs)

    def add_pg_loss(self, advantages):
        '''
        Compute loss based on the policy gradient theorem.

        Parameters
        ----------
        batch: dict
            The batch should contain all the information necessary
            to compute the gradients.
        '''
        log_probs = torch.cat(self.saved_log_probs).view(-1)
        objective = log_probs * advantages
        loss = -objective.sum()

        self.losses.append(loss)

    def add_value_nn_loss(self, vtarget):
        state_values = torch.cat(self.saved_state_values).view(-1)

        loss = F.mse_loss(input=state_values, target=vtarget)

        self.losses.append(loss)

    def add_losses(self, batch):
        '''
        Define all losses used for calculating the gradient.

        Parameters
        ----------
        batch: dict
            The batch should contain all the information necessary
            to compute the gradients.
        '''
        advantages = self._to_variable(batch['advantages'])
        vtarget = self._to_variable(batch['vtarget'])

        self.add_pg_loss(advantages)
        self.add_value_nn_loss(vtarget)

        self.saved_log_probs = []
        self.saved_state_values = []
