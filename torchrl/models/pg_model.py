import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

# from torchrl.distributions import CategoricalDist, NormalDist
import torchrl.utils as U
from torchrl.models import BaseModel


class PGModel(BaseModel):
    def __init__(self, policy_nn_config, value_nn_config=None, share_body=False,
                 **kwargs):
        self.policy_nn_config = policy_nn_config
        self.value_nn_config = value_nn_config
        self.share_body = share_body
        self.saved_dists = []

        super().__init__(**kwargs)

    def create_networks(self):
        self.policy_nn = self.net_from_config(self.policy_nn_config)

        if self.value_nn_config is not None:
            body = self.policy_nn.body if self.share_body else None
            self.value_nn = self.net_from_config(self.value_nn_config, body=body)
        else:
            self.value_nn = None

    def create_dist(self, parameters):
        if self.action_info['dtype'] == 'discrete':
            logits = parameters
            return Categorical(logits=logits)

        elif self.action_info['dtype'] == 'continuous':
            means = parameters[:, 0]
            std_devs = parameters[:, 1].exp()

            return Normal(loc=means, scale=std_devs)

        else:
            raise ValueError('No distribution is defined for {} actions'.format(
                self.action_info['dtype']))

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
        x = self.policy_nn.head(self.policy_nn.body(x))
        # action_probs = F.softmax(action_scores, dim=1)

        # return action_probs
        return x

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
        parameters = self.forward(state)
        dist = self.create_dist(parameters[0])
        action = dist.sample()
        self.saved_dists.append(dist)

        return U.to_numpy(action)

    def add_losses(self, batch):
        '''
        Define all losses used for calculating the gradient.

        Parameters
        ----------
        batch: dict
            The batch should contain all the information necessary
            to compute the gradients.
        '''
        self.add_pg_loss(batch)
        self.add_value_nn_loss(batch)

    def add_value_nn_loss(self, batch):
        state_values = self.value_nn.head(self.value_nn.body(batch['state_ts']))
        vtarget = self._to_variable(batch['vtarget'])

        loss = F.mse_loss(input=state_values.view(-1), target=vtarget)
        self.losses.append(loss)

        # Add logs
        self.logger.add_log('Loss/value_nn/mse', loss.item())
        ev = 1 - torch.var(vtarget - state_values.view(-1)) / torch.var(vtarget)
        self.logger.add_log('Value_NN/explained_variance', ev.item())
        # self.logger.add_log('vtarget_var', torch.var(vtarget).item())
        # self.logger.add_log('value_nn_var', torch.var(state_values).item())

    def add_state_values(self, traj):
        if self.value_nn is not None:
            state_values = self.value_nn.head(self.value_nn.body(traj['state_ts']))
            state_values = state_values.data.view(-1).cpu().numpy()
            traj['state_values'] = state_values
        else:
            pass

    def entropy(self):
        entropies = [dist.entropy() for dist in self.saved_dists]
        return torch.cat(entropies).mean()

    def write_logs(self, batch):
        self.logger.add_log('Policy/Entropy', self.entropy().data[0])
