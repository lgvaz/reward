import reward.utils as U
from .base_transform import BaseTransform


class StackStates(BaseTransform):
    def __init__(self, n, dim=1):
        super().__init__()
        self.n = n
        self.dim = dim
        self.ring_buffer = None
        self.eval_ring_buffer = None

        if dim != 1:
            err_msg = (
                "Because of the way of the way the rbuff is currently"
                "implemented, we're only allowed to stack in the first dimension"
                "(which should be the case for image stacking). Support for other"
                "options will be added in the future"
            )
            raise ValueError(err_msg)

    def transform(self, s):
        s = U.to_np(s)
        assert (
            s.shape[self.dim + 1] == 1
        ), "Dimension to stack must be 1 but it is {}".format(s.shape[self.dim + 1])

        return s.swapaxes(0, self.dim + 1)[0]

    def transform_s(self, s, training=True):
        if self.ring_buffer is None:
            self.ring_buffer = U.buffers.RingBuffer(input_shape=s.shape, maxlen=self.n)
        if self.eval_ring_buffer is None:
            # First dimension (num_envs) for evaluation is always 1
            eval_shape = (1,) + s.shape[1:]
            self.eval_ring_buffer = U.buffers.RingBuffer(
                input_shape=eval_shape, maxlen=self.n
            )

        if training:
            self.ring_buffer.append(s)
            s = self.ring_buffer.get_data()
        else:
            self.eval_ring_buffer.append(s)
            s = self.eval_ring_buffer.get_data()

        return self.transform(s)


class StateRunNorm(BaseTransform):
    def __init__(self, clip_range=5):
        super().__init__()
        self.filt = None
        self.clip_range = clip_range

    def transform_s(self, s, training=True):
        if self.filt is None:
            shape = s.shape
            if len(shape) != 2:
                msg = "state shape must (num_envs, num_features, got {})".format(shape)
                raise ValueError(msg)
            self.filt = U.filter.MeanStdFilter(
                num_features=s.shape[-1], clip_range=self.clip_range
            )

        s = self.filt.normalize(s, add_sample=training)
        return s

    def transform_batch(self, batch, training=True):
        if training:
            self.filt.update()
        return batch

    def write_logs(self, logger):
        logger.add_tf_only_log("Env/State/mean", self.filt.mean.mean())
        logger.add_tf_only_log("Env/State/std", self.filt.std.mean())


class Frame2Float(BaseTransform):
    def transform_s(self, s, training=True):
        return s.astype("float32") / 255.0
