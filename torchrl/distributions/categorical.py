from torchrl.distributions_temp.kl import register_kl
from torchrl.distributions_temp import Categorical
from torchrl.utils import EPSILON


class CategoricalDist(Categorical):
    @property
    def log_probs(self):
        return self.probs.clamp(min=EPSILON, max=1 - EPSILON).log()

    def detach(self):
        self.__dict__ = dict((key, value.detach()) if hasattr(value, 'detach') else (
            key, value) for key, value in self.__dict__.items())

    # def entropy(self):
    #     return -(self.probs * self.log_probs).sum(-1)

    # @staticmethod
    # def kl_divergence(old_dist, new_dist):
    #     assert isinstance(old_dist, CategoricalDist) and isinstance(
    #         new_dist, CategoricalDist)

    #     return (old_dist.probs * (old_dist.log_probs - new_dist.log_probs)).sum(-1)


@register_kl(CategoricalDist, CategoricalDist)
def kl_cat_cat(p, q):
    return (p.probs * (p.log_probs - q.log_probs)).sum(-1)
