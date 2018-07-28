import torchrl.utils as U
from .base_transform import BaseTransform


class StackStates(BaseTransform):
    def __init__(self, n, dim=1):
        super().__init__()
        self.n = n
        self.dim = dim
        self.ring_buffer = None
        self.eval_ring_buffer = None

    def transform(self, state):
        state = U.to_np(state)
        assert (
            state.shape[self.dim + 1] == 1
        ), "Dimension to stack must be 1 but it is {}".format(
            state.shape[self.dim + 1]
        )

        return state.swapaxes(0, self.dim + 1)[0]

    def transform_state(self, state, training=True):
        if self.ring_buffer is None:
            self.ring_buffer = U.buffers.RingBuffer(
                input_shape=state.shape, maxlen=self.n
            )
        if self.eval_ring_buffer is None:
            # First dimension (num_envs) for evaluation is always 1
            eval_shape = (1,) + state.shape[1:]
            self.eval_ring_buffer = U.buffers.RingBuffer(
                input_shape=eval_shape, maxlen=self.n
            )

        if training:
            self.ring_buffer.append(state)
            state = self.ring_buffer.get_data()
        else:
            self.eval_ring_buffer.append(state)
            state = self.eval_ring_buffer.get_data()

        return U.LazyArray(state, transform=self.transform)


class StateRunNorm(BaseTransform):
    def __init__(self, clip_range=5, use_latest_filter_update=False):
        super().__init__()
        self.filt = None
        self.clip_range = clip_range
        self.use_latest_filter_update = use_latest_filter_update

    def transform_state(self, state, training=True):
        if self.filt is None:
            shape = state.shape
            if len(shape) != 2:
                raise ValueError(
                    "state shape must be in the form (num_envs, num_features), got {}".format(
                        shape
                    )
                )
            self.filt = U.filters.MeanStdFilter(
                num_features=state.shape[-1], clip_range=self.clip_range
            )

        state = self.filt.normalize(
            state, add_sample=training, use_latest_update=self.use_latest_filter_update
        )
        return state

    def transform_batch(self, batch, training=True):
        if training:
            self.filt.update()
        return batch

    def write_logs(self, logger):
        logger.add_tf_only_log("Env/State/mean", self.filt.mean.mean())
        logger.add_tf_only_log("Env/State/std", self.filt.std.mean())


class Frame2Float(BaseTransform):
    def transform_state(self, state, training=True):
        return U.LazyArray(data=state, transform=lambda x: x.astype("float") / 255.)
