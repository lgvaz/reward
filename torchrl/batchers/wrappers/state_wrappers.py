import numpy as np
from torchrl.utils import LazyArray, to_np
from torchrl.batchers.wrappers import BaseWrapper
from torchrl.utils.filters import MeanStdFilter
from torchrl.utils.buffers import RingBuffer


class StateRunNorm(BaseWrapper):
    def __init__(self, batcher):
        super().__init__(batcher=batcher)
        shape = self.get_state_info().shape
        assert len(shape) == 1
        self.filt = MeanStdFilter(num_features=shape[0])

    def transform_state(self, state, training=True):
        state = self.filt.normalize(state, add_sample=training)
        return self.old_transform_state(state, training=training)

    def transform_batch(self, batch, training=True):
        if training:
            self.filt.update()
        return self.old_transform_batch(batch, training=training)

    def write_logs(self, logger):
        logger.add_tf_only_log('Env/State/mean', self.filt.mean.mean())
        logger.add_tf_only_log('Env/State/std', self.filt.std.mean())

        self.old_write_logs(logger)


# class StackFrames(BaseWrapper):
#     def __init__(self, batcher, stack_frames=4):
#         super().__init__(batcher=batcher)
#         self.stack_frames = stack_frames
#         self.ring_buffer = RingBuffer(
#             input_shape=self.runner.get_state_info().shape, maxlen=self.stack_frames)
#         # First dimension (num_envs) for evaluation is always 1
#         eval_in_shape = (1, ) + self.runner.get_state_info().shape[1:]
#         self.eval_ring_buffer = RingBuffer(
#             input_shape=eval_in_shape, maxlen=self.stack_frames)

#     def transform_state(self, state, training=True):
#         if training:
#             self.ring_buffer.append(state)
#             state = self.ring_buffer.get_data()
#         else:
#             self.eval_ring_buffer.append(state)
#             state = self.eval_ring_buffer.get_data()

#         return self.old_transform_state(state, training=training)


class StackFrames(BaseWrapper):
    def __init__(self, batcher, stack_frames=4, dim=1):
        super().__init__(batcher=batcher)
        self.n = stack_frames
        self.dim = dim
        self.ring_buffer = None
        self.eval_ring_buffer = None

    def transform(self, state):
        state = to_np(state)
        assert state.shape[self.dim
                           + 1] == 1, 'Dimension to stack must be 1 but it is {}'.format(
                               state.shape[self.dim + 1])

        return state.swapaxes(0, self.dim + 1)[0]

    def transform_state(self, state, training=True):
        if self.ring_buffer is None:
            self.ring_buffer = RingBuffer(input_shape=state.shape, maxlen=self.n)
        if self.eval_ring_buffer is None:
            # First dimension (num_envs) for evaluation is always 1
            eval_shape = (1, ) + state.shape[1:]
            self.eval_ring_buffer = RingBuffer(input_shape=eval_shape, maxlen=self.n)

        if training:
            self.ring_buffer.append(state)
            state = self.ring_buffer.get_data()
        else:
            self.eval_ring_buffer.append(state)
            state = self.eval_ring_buffer.get_data()

        state = LazyArray(state, transform=self.transform)

        return self.old_transform_state(state, training=training)


class Frame2Float(BaseWrapper):
    def transform_state(self, state, training=True):
        state = state.astype('float32') / 255.

        return self.old_transform_state(state, training=training)

    # def transform_state(self, state, training=True):
    #     state = LazyArray(state)

    #     return self.old_transform_state(state, training=training)
