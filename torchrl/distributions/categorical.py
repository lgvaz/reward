import torch
from torch.distributions import kl


class Categorical(torch.distributions.Categorical):
    def log_prob(self, value):
        return super().log_prob(value)[..., None]


@kl.register_kl(Categorical, Categorical)
def _kl_categorical_categorical(p, q):
    return kl._kl_categorical_categorical(p, q)[..., None]
