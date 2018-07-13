from torchrl.batchers.wrappers import BaseWrapper
from torchrl.utils.filters import MeanStdFilter


class RewardRunScaler(BaseWrapper):
    def __init__(self, batcher):
        super().__init__(batcher=batcher)
        self.filt = MeanStdFilter(shape=(1, ))

    def transform_batch(self, batch, training=True):
        batch.reward = self.filt.scale(batch.reward, add_sample=training)
        if training:
            self.filt.update()

        return self.old_transform_batch(batch, training=training)

    def write_logs(self, logger):
        logger.add_tf_only_log('Env/Reward/mean', self.filt.mean.mean())
        logger.add_tf_only_log('Env/Reward/std', self.filt.std.mean())

        self.old_write_logs(logger)


class RewardClipper(BaseWrapper):
    def __init__(self, batcher, clip_range=1.):
        super().__init__(batcher=batcher)
        self.clip_range = clip_range
        self.rewards = None

    def transform_batch(self, batch, training=True):
        batch.reward = batch.reward.clip(min=-self.clip_range, max=self.clip_range)
        self.rewards = batch.reward

        return self.old_transform_batch(batch, training=training)
