import torch
import reward.utils as U
from reward.distributions import Normal


class TanhNormal(Normal):
    def log_prob(self, value):
        raise ValueError("For numerical stability you shoud use `log_prob_pre`")

    def log_prob_pre(self, pre_tanh):
        log_prob_pre = super().log_prob(pre_tanh)
        after_tanh = torch.tanh(pre_tanh)
        log_prob = log_prob_pre - (1 - after_tanh.pow(2) + U.EPSILON).log()
        return log_prob

    def sample_with_pre(self, sample_shape=torch.Size()):
        value = super().sample(sample_shape=sample_shape)
        return torch.tanh(value), value

    def sample(self, sample_shape=torch.Size()):
        return self.sample_with_pre(sample_shape=sample_shape)[0]

    def rsample_with_pre(self, sample_shape=torch.Size()):
        value = super().rsample(sample_shape=sample_shape)
        return torch.tanh(value), value

    def rsample(self, sample_shape=torch.Size()):
        return self.rsample_with_pre(sample_shape=sample_shape)[0]

    def sample_det(self):
        return torch.tanh(self.loc)
