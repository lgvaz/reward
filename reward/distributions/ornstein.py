import torch
import numpy as np
from reward.distributions import BaseDist


# TODO: REFACTOR
class Ornstein(BaseDist):
    """
    From `Baselines <https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py#L49>`_.
    """

    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __getitem__(self, key):
        return Ornstein(
            mu=self.mu[key],
            sigma=self.sigma[key],
            theta=self.theta,
            dt=self.dt,
            x0=self.x0,
        )

    def sample(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return "OrnsteinUhlenbeckActionNoise(mu={}, sigma={})".format(
            self.mu, self.sigma
        )
