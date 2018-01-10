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
