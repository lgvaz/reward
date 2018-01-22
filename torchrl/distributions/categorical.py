from torch.distributions import Categorical
from torchrl.utils import EPSILON


class CategoricalDist(Categorical):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.last_action = None

    @property
    def log_probs(self):
        return self.probs.clamp(min=EPSILON, max=1 - EPSILON).log()

    @property
    def last_log_prob(self):
        return self.log_prob(self.last_action)

    def sample(self, *args, **kwargs):
        action = super().sample(*args, **kwargs)
        self.last_action = action

        return action

    def entropy(self):
        return -(self.probs * self.log_probs).sum(-1)

    def detach(self):
        self.__dict__ = dict((key, value.detach()) if hasattr(value, 'detach') else (
            key, value) for key, value in self.__dict__.items())

    @staticmethod
    def kl_divergence(old_dist, new_dist):
        assert isinstance(old_dist, CategoricalDist) and isinstance(
            new_dist, CategoricalDist)

        return (old_dist.probs * (old_dist.log_probs - new_dist.log_probs)).sum(-1)
