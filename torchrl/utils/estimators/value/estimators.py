from torchrl.utils.estimators.estimation_funcs import td_target


class CompleteReturn:
    def __call__(self, batch):
        return batch.return_


class TDTarget:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, batch):
        return td_target(
            rewards=batch.reward,
            dones=batch.done,
            state_values=batch.state_value,
            gamma=self.gamma)


# TODO: Not really GAE estimation... Only gae in policy too
class GAE:
    def __call__(self, batch):
        return batch.advantage + batch.state_value
