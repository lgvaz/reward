import torch
from torch.distributions import Categorical, Normal
import torchrl.utils as U
from torchrl.models import BaseModel


class BasePGModel(BaseModel):
    '''
    Base class for all Policy Gradient Models.
    '''

    def select_action(self, state):
        '''
        Define how the actions are selected, in this case the actions
        are sampled from a distribution which values are given be a NN.

        Parameters
        ----------
        state: np.array
            The state of the environment (can be a batch of states).
        '''
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
        parameters: np.array
        The parameters are used to create a distribution
        (continuous or discrete depending on the type of the environment).
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
