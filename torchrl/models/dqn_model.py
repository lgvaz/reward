from copy import deepcopy
from torchrl.models import QModel


class DQNModel(QModel):
    def __init__(self, model, batcher, exploration_rate, target_up_freq, **kwargs):
        super().__init__(
            model=model, batcher=batcher, exploration_rate=exploration_rate, **kwargs
        )
        self.target_up_freq = target_up_freq

        self.target_net = deepcopy(self.model)
        self.target_net.eval()

        import pdb

        pdb.set_trace()
