import torch
from torch.distributions import Categorical, Normal
import torchrl.utils as U
from torchrl.models import BaseModel


class BasePGModel(BaseModel):
    '''
    Base class for all Policy Gradient Models.

    Parameters
    ----------
    model: torch model
        A pytorch neural network.
    action_info: dict
        Dict containing information about the action space.
    '''

    def __init__(self, model, action_info, **kwargs):
        super().__init__(model, **kwargs)
        self.action_info = action_info

    def select_action(self, state):
        parameters = self.forward(state)
        dist = self.create_dist(parameters[0])
        self.memory.dists.append(dist)
        action = dist.sample()

        return U.to_numpy(action)

    def create_dist(self, parameters):
        '''
        Specify how the policy distributions should be created.
        The type of the distribution depends on the environment.

        Parameters
        ----------
        parameters
        '''
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

    def extract_log_probs(self, actions, dists):
        return torch.stack([d.log_prob(a).sum() for d, a in zip(dists, actions)])
