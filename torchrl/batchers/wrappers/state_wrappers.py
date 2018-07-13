from torchrl.batchers.wrappers import BaseWrapper
from torchrl.utils.filters import MeanStdFilter
from torchrl.utils.memories import RingBuffer


class StateRunNorm(BaseWrapper):
    def __init__(self, batcher):
        super().__init__(batcher=batcher)
        self.filt = MeanStdFilter(shape=self.get_state_info().shape)

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


class StackFrames(BaseWrapper):
    def __init__(self, batcher, stack_frames=4):
        super().__init__(batcher=batcher)
        self.stack_frames = stack_frames
        self.ring_buffer = RingBuffer(
            input_shape=self.runner.get_state_info().shape, maxlen=self.stack_frames)
        # First dimension (num_envs) for evaluation is always 1
        eval_in_shape = (1, ) + self.runner.get_state_info().shape[1:]
        self.eval_ring_buffer = RingBuffer(
            input_shape=eval_in_shape, maxlen=self.stack_frames)

    def transform_state(self, state, training=True):
        if training:
            self.ring_buffer.append(state)
            state = self.ring_buffer.get_data()
        else:
            self.eval_ring_buffer.append(state)
            state = self.eval_ring_buffer.get_data()

        return self.old_transform_state(state, training=training)


class Frame2Float(BaseWrapper):
    def transform_state(self, state, training=True):
        state = state.astype('float32') / 255.

        return self.old_transform_state(state, training=training)
