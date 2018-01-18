from torch.distributions import Categorical
from torchrl.utils import EPSILON


class CategoricalDist(Categorical):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.last_value = None

    @property
    def last_log_prob(self):
        return self.log_prob(self.last_value)

    def sample(self, *args, **kwargs):
        value = super().sample(*args, **kwargs)
        self.last_value = value

        return value

    def entropy(self):
        log_probs = self.probs.clamp(min=EPSILON, max=1 - EPSILON).log()
        return -(self.probs * log_probs).sum(-1)
