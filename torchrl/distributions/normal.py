import torch
from torch.distributions import kl
from torchrl.distributions import BaseDist


class Normal(torch.distributions.Normal, BaseDist):
    def __getitem__(self, key):
        return Normal(loc=self.loc[key], scale=self.scale[key])


@kl.register_kl(Normal, Normal)
def _kl_normal_normal(p, q):
    return kl._kl_normal_normal(p, q)
