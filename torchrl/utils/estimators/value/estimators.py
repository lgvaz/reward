from torchrl.utils.estimators.estimation_funcs import td_target
from torchrl.utils.estimators import BaseEstimator


class CompleteReturn(BaseEstimator):
    def __call__(self, batch):
        return batch.return_


class TDTarget(BaseEstimator):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, batch):
        return td_target(
            rewards=batch.reward,
            dones=batch.done,
            state_values=batch.state_value,
            gamma=self.gamma)


# TODO: Not really GAE estimation... Only gae in policy too
class GAE(BaseEstimator):
    def __call__(self, batch):
        return batch.advantage + batch.state_value
