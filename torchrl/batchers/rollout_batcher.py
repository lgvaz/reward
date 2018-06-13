import numpy as np
import torchrl.utils as U
from torchrl.batchers import BaseBatcher


class RolloutBatcher(BaseBatcher):
    def get_batch(self, select_action_fn):
        super().get_batch(select_action_fn=select_action_fn)
        horizon = self.batch_size // self.env.num_envs
        batch = U.Batch()

        for i in range(horizon):
            action = select_action_fn(self._state_t)

            # TODO: Add info to batch
            state_tp1, reward, done, info = self.env.step(action)
            state_tp1 = self.transform_state(state_tp1)

            batch.state_t_and_tp1.append(self._state_t)
            batch.action.append(action)
            batch.reward.append(reward)
            batch.done.append(done)

            self._state_t = state_tp1

        # batch.state_t_and_tp1.append(state_tp1)
        batch.state_t_and_tp1.append(self._state_t)
        batch = batch.apply_to_all(np.array)

        batch.state_t = batch.state_t_and_tp1[:-1]
        batch.state_tp1 = batch.state_t_and_tp1[1:]

        return batch
