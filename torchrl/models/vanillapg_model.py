import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from torchrl.models import BaseModel

from torch.autograd import Variable

from torchrl.utils import discounted_sum_rewards
from torch.autograd import Variable
import numpy as np


class VanillaPGModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.selected_log_probs = []

    def forward(self, x):
        action_scores = super().forward(x)
        return F.softmax(action_scores, dim=1)

    def select_action(self, state):
        probs = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        # action = Variable(torch.zeros((1)).long().cuda())
        log_prob = dist.log_prob(action)
        self.selected_log_probs.append(log_prob)

        return action

    def train(self, batch):
        # TODO: remove hardcoded cuda
        returns = torch.Tensor(batch['returns'])
        # returns = (returns - returns.mean()) / (returns.std() - 1e-6)
        objective = torch.stack(self.selected_log_probs) * Variable(
            returns[:, None]).cuda()

        self.opt.zero_grad()
        loss = -objective.sum()
        loss.backward()
        self.opt.step()

        self.selected_log_probs = []

    # def train(self, rewards):
    #     returns = discounted_sum_rewards(rewards)
    #     # returns = (returns - returns.mean()) / (returns.std() + 1e-7)
    #     objective = torch.stack(self.selected_log_probs) * Variable(
    #         torch.Tensor(returns))[:, None].cuda()

    #     self.opt.zero_grad()
    #     loss = -objective.sum()
    #     loss.backward()
    #     self.opt.step()

    #     self.selected_log_probs = []
