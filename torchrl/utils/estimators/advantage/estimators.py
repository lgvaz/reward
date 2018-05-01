from torchrl.utils.estimators import BaseEstimator
from torchrl.utils.estimators.estimation_funcs import (discounted_sum_rewards,
                                                       gae_estimation, td_target)


class CompleteReturn(BaseEstimator):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, batch):
        return discounted_sum_rewards(
            rewards=batch.rewards,
            dones=batch.done,
            last_state_value=batch.state_value[-1],
            gamma=self.gamma)


class Baseline(BaseEstimator):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, batch):
        return_ = discounted_sum_rewards(
            rewards=batch.rewards,
            dones=batch.done,
            last_state_value=batch.state_value[-1],
            gamma=self.gamma)
        return return_ - batch.state_value


class TD(BaseEstimator):
    def __init__(self, gamma=0.99):
        self.gamma = gamma

    def __call__(self, batch):
        return td_target(
            rewards=batch.reward,
            dones=batch.done,
            state_values=batch.state_value,
            gamma=self.gamma) - batch.state_value


class GAE(BaseEstimator):
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def __call__(self, batch):
        return gae_estimation(
            rewards=batch.reward,
            dones=batch.done,
            state_values=batch.state_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda)
