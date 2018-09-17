import pdb
import torch
import reward.utils as U
from reward.distributions import Normal


class TanhNormal(Normal):
    def __init__(self, *args, **kwargs):
        # assert not torch.isnan(kwargs["loc"]).any()
        super().__init__(*args, **kwargs)

    def _atanh(self, value):
        return 0.5 * ((1 + value) / (1 - value)).log()

    def log_prob(self, value):
        pre_tanh_value = self._atanh(value=value)
        pre_log_prob = super().log_prob(value=pre_tanh_value)
        log_prob = pre_log_prob - (1 - value.pow(2) + U.EPSILON).log()
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
