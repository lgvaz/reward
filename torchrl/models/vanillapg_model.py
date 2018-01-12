import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from torchrl.models import BaseModel
from torchrl.utils import discounted_sum_rewards


class VanillaPGModel(BaseModel):
    '''
    Vanilla Policy Gradient model.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.saved_log_probs = []

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
        action_scores = self.nn_head.main(self.nn_body.main(x))
        return F.softmax(action_scores, dim=1)

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
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.saved_log_probs.append(log_prob)

        return action.data[0]

    def train(self, batch):
        '''
        Compute and apply gradients based on the policy gradient theorem.

        Should use the batch to compute and apply gradients to the network.

        Parameters
        ----------
        batch: dict
            The batch should contain all the information necessary
            to compute the gradients.
        '''
        super().train()
        returns = discounted_sum_rewards(batch['rewards'])
        returns = Variable(self._maybe_cuda(torch.from_numpy(returns).float()))
        objective = torch.cat(self.saved_log_probs) * returns

        self.opt.zero_grad()
        loss = -objective.sum()
        loss.backward()
        self.opt.step()

        self.saved_log_probs = []
