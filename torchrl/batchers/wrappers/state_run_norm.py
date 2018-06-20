from torchrl.batchers.wrappers import BaseWrapper
from torchrl.utils.filters import MeanStdFilter


class StateRunNorm(BaseWrapper):
    def __init__(self, batcher):
        super().__init__(batcher=batcher)
        self.filt = MeanStdFilter(shape=self.get_state_info().shape)

    def transform_state(self, state):
        state = self.filt.normalize(state)
        return self.old_transform_state(state)

    def transform_batch(self, batch):
        self.filt.update()
        return self.old_transform_batch(batch)
