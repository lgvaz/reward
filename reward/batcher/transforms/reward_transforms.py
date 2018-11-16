import reward.utils as U
from .base_transform import BaseTransform


class RewardConstScaler(BaseTransform):
    def __init__(self, factor=0.01):
        super().__init__()
        self.factor = factor

    def transform_batch(self, batch, training=True):
        batch.r = batch.r * self.factor

        return batch


class RewardRunScaler(BaseTransform):
    def __init__(self):
        super().__init__()
        self.filt = U.filter.MeanStdFilter(num_features=1)

    def transform_batch(self, batch, training=True):
        rew_shape = batch.r.shape
        if len(rew_shape) != 2:
            raise ValueError(
                "Reward shape should be in the form (num_steps, num_envs) and is {}".format(
                    rew_shape
                )
            )

        batch.r = self.filt.scale(batch.r.reshape(-1, 1), add_sample=training)

        if training:
            self.filt.update()

        batch.r = U.to_np(batch.r).reshape(rew_shape)

        return batch

    def write_logs(self, logger):
        logger.add_tf_only_log("Env/Reward/mean", self.filt.mean.mean())
        logger.add_tf_only_log("Env/Reward/std", self.filt.std.mean())


class RewardClipper(BaseTransform):
    def __init__(self, clip_range=1.):
        super().__init__()
        self.clip_range = clip_range
        self.rs = None

    def transform_batch(self, batch, training=True):
        batch.r = batch.r.clip(min=-self.clip_range, max=self.clip_range)
        self.rs = batch.r

        return batch
