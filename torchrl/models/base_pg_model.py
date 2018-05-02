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

    def __init__(self, model, env, **kwargs):
        super().__init__(model, env, **kwargs)

    def select_action(self, state):
        parameters = self.forward(state)
        dist = self.create_dist(parameters)
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
        if self.env.action_info['dtype'] == 'discrete':
            logits = parameters
            return Categorical(logits=logits)

        elif self.env.action_info['dtype'] == 'continuous':
            means = parameters[..., 0]
            std_devs = parameters[..., 1].exp()

            return Normal(loc=means, scale=std_devs)

        else:
            raise ValueError('No distribution is defined for {} actions'.format(
                self.env.action_info['dtype']))
