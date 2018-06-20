from torchrl.batchers.wrappers import BaseWrapper
from torchrl.utils.filters import MeanStdFilter


class RewardRunScaler(BaseWrapper):
    def __init__(self, batcher):
        super().__init__(batcher=batcher)
        self.filt = MeanStdFilter(shape=(1, ))

    def transform_batch(self, batch):
        batch.reward = self.filt.scale(batch.reward)
        self.filt.update()
        return self.old_transform_batch(batch)
