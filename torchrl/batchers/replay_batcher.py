import numpy as np
import torchrl.utils as U
from torchrl.batchers import BaseBatcher


class ReplayBatcher(BaseBatcher):
    def __init__(self, runner, *, batch_size, steps_per_batch, replay_buffer=None):
        super().__init__(runner=runner, batch_size=batch_size)
        self.steps_per_batch = steps_per_batch
        self.replay_buffer = replay_buffer or U.buffers.ReplayBuffer(maxlen=int(1e6))

    def get_batch(self, select_action_fn):
        super().get_batch(select_action_fn=select_action_fn)

        for i in range(self.steps_per_batch):
            action = select_action_fn(self._state_t, self.num_steps)

            state_tp1, reward, done, info = self.runner.act(action)
            import pdb

            pdb.set_trace()
            state_tp1 = self.transform_state(state_tp1)

            self.replay_buffer.add_sample()

    # def transform_state(self, state, training=True):
    #     import pdb
    #     pdb.set_trace()
