import torch
from torch.distributions import kl
from torchrl.distributions import BaseDist


class Categorical(torch.distributions.Categorical, BaseDist):
    def __getitem__(self, key):
        return Categorical(logits=self.logits[key])

    def log_prob(self, value):
        return super().log_prob(value)[..., None]


@kl.register_kl(Categorical, Categorical)
def _kl_categorical_categorical(p, q):
    return kl._kl_categorical_categorical(p, q)[..., None]
