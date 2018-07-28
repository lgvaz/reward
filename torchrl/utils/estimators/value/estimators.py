from torchrl.utils.estimators import BaseEstimator
from torchrl.utils.estimators.estimation_funcs import discounted_sum_rewards, td_target


class CompleteReturn(BaseEstimator):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, batch):
        return discounted_sum_rewards(
            rewards=batch.reward,
            dones=batch.done,
            last_state_value_t=batch.state_value_t[-1],
            gamma=self.gamma,
        )


class TDTarget(BaseEstimator):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, batch):
        return td_target(
            rewards=batch.reward,
            dones=batch.done,
            state_value_tp1=batch.state_value_tp1,
            gamma=self.gamma,
        )


class FromAdvantage(BaseEstimator):
    def __call__(self, batch):
        return batch.advantage + batch.state_value_t
