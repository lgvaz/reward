import torch


class Categorical(torch.distributions.Categorical):
    def log_prob(self, value):
        return super().log_prob(value)[..., None]
