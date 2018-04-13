import torch
from torch.distributions import Categorical, Normal
import torchrl.utils as U
from torchrl.models import BaseModel


class PGModel(BaseModel):
    def __init__(self, model, action_info, **kwargs):
        super().__init__(model, **kwargs)
        self.saved_dists = []
        self.action_info = action_info

    def select_action(self, state):
        parameters = self.forward(state)
        dist = self.create_dist(parameters[0])
        self.saved_dists.append(dist)
        action = dist.sample()

        return U.to_numpy(action)

    def add_losses(self, batch):
        self.pg_loss(batch)

    def pg_loss(self, batch):
        objective = batch.log_probs * batch.advantages
        loss = -objective.mean()

        self.losses.append(loss)

    def train(self, batch):
        batch = batch.apply_to_all(self._to_tensor)

        batch.log_probs = torch.stack([
            dist.log_prob(action).sum()
            for dist, action in zip(self.saved_dists, batch.actions)
        ])

        self.optimizer_step(batch)

        self.saved_dists = []

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
