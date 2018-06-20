from torchrl.batchers.wrappers import BaseWrapper
from torchrl.utils.memories import RingBuffer


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
