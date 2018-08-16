from torchrl.utils.estimators import BaseEstimator
from torchrl.utils.estimators.estimation_funcs import q_learning_target


class QLearningTarget(BaseEstimator):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, batch):
        return q_learning_target(
            rewards=batch.reward, dones=batch.done, q_tp1=batch.q_tp1, gamma=self.gamma
        )
