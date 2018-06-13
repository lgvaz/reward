import numpy as np
from torchrl.batchers import BaseBatcher, RolloutBatcher
from torchrl.utils.memories import RingBuffer


class ImgBatcher(BaseBatcher):
    def __init__(self, env, *, batch_size, stack_frames=4, normalize_frames=True):
        super().__init__(env=env, batch_size=batch_size)
        self.normalize_frames = normalize_frames
        # TODO: get_shape
        self.ring_buffer = RingBuffer(
            input_shape=(self.env.num_envs, ) + self.env.get_state_info().shape,
            maxlen=stack_frames)

    def transform_state(self, state):
        state = state.astype(np.float32)
        if self.normalize_frames:
            state /= 255.
        self.ring_buffer.append(state)
        state = self.ring_buffer.get_data()

        return state


class ImgRolloutBatcher(ImgBatcher, RolloutBatcher):
    pass
