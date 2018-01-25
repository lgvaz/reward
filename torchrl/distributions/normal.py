from torchrl.distributions_temp import Normal
from torchrl.utils import EPSILON


class NormalDist(Normal):
    # def __init__(self, mean_and_log_std):
    #     mean, log_std = mean_and_log_std
    #     super().__init__(loc=mean, scale=log_std.exp())

    @property
    def log_probs(self):
        return self.probs.clamp(min=EPSILON, max=1 - EPSILON).log()

    def detach(self):
        self.__dict__ = dict((key, value.detach()) if hasattr(value, 'detach') else (
            key, value) for key, value in self.__dict__.items())
