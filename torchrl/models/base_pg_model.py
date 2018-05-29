from abc import abstractproperty
import torch
from torchrl.distributions import Categorical, Normal
import torchrl.utils as U
from torchrl.models import BaseModel
from torchrl.nn import ActionLinear


class BasePGModel(BaseModel):
    '''
    Base class for all Policy Gradient Models.
    '''

    def __init__(self, model, env, *, entropy_coef=0, **kwargs):
        super().__init__(model=model, env=env, **kwargs)
        self.entropy_coef_fn = U.make_callable(entropy_coef)

    @abstractproperty
    def entropy(self):
        pass

    @property
    def entropy_coef(self):
        return self.entropy_coef_fn(self.step)

    def entropy_loss(self, batch):
        '''
        Adds a entropy cost to the loss function,
        with the intent of encouraging exploration.

        Parameters
        ----------
        batch: Batch
            The batch should contain all the information necessary
            to compute the gradients.
        '''
        loss = -self.entropy * self.entropy_coef
        return loss

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

        return U.to_np(action)

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

    def write_logs(self, batch):
        super().write_logs(batch)
        self.add_log('Entropy', self.entropy)

    @staticmethod
    def output_layer(input_shape, action_info):
        return ActionLinear(in_features=input_shape, action_info=action_info)
