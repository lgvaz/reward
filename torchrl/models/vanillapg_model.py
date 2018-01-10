import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from torchrl.models import BaseModel
from torchrl.utils import discounted_sum_rewards


class VanillaPGModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.saved_log_probs = []

    def forward(self, x):
        action_scores = super().forward(x)
        return F.softmax(action_scores, dim=1)

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     import torch.nn as nn
    #     self.model1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU()).cuda()
    #     self.model2 = nn.Sequential(nn.Linear(64, 2)).cuda()
    #     self.saved_log_probs = []
    #     self.opt = self._create_optimizer()

    # def forward(self, x):
    #     from torch.autograd import Variable
    #     action_scores = self.model2(
    #         self.model1(Variable(torch.from_numpy(x).float().cuda())))
    #     return F.softmax(action_scores, dim=1)

    def select_action(self, state):
        probs = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.saved_log_probs.append(log_prob)

        return action.data[0]

    def train(self, batch):
        returns = discounted_sum_rewards(batch['rewards'])
        objective = torch.cat(self.saved_log_probs) * torch.autograd.Variable(
            torch.from_numpy(returns).float().cuda())

        self.opt.zero_grad()
        loss = -objective.sum()
        loss.backward()
        self.opt.step()

        self.saved_log_probs = []
