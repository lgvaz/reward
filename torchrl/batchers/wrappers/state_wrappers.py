from torchrl.batchers.wrappers import BaseWrapper
from torchrl.utils.filters import MeanStdFilter
from torchrl.utils.memories import RingBuffer


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

    def transform_state(self, state):
        self.ring_buffer.append(state)
        state = self.ring_buffer.get_data()

        return self.old_transform_state(state)


class Frame2Float(BaseWrapper):
    def transform_state(self, state):
        state = state.astype('float32') / 255.

        return self.old_transform_state(state)
