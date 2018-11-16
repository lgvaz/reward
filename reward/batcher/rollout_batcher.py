import torch
import numpy as np
import reward.utils as U
from reward.batcher import BaseBatcher


class RolloutBatcher(BaseBatcher):
    def get_batch(self, act_fn):
        if self.state_t is None:
            self.state_t = self.transform_state(self.runner.reset())
            self.state_t = U.to_tensor(self.state_t)

        horizon = self.batch_size // self.runner.num_envs
        batch = U.Batch(initial_keys=["state_t_and_tp1", "action", "reward", "done"])

        for i in range(horizon):
            action = act_fn(self.state_t, self.num_steps)

            sn, reward, done, info = self.runner.act(action)
            sn = U.to_tensor(self.transform_state(sn))

            batch.state_t_and_tp1.append(self.state_t)
            batch.action.append(action)
            batch.reward.append(reward)
            batch.done.append(done)
            # batch.info.append(info)

            self.state_t = sn
        batch.state_t_and_tp1.append(self.state_t)

        batch.state_t_and_tp1 = torch.stack(batch.state_t_and_tp1)
        batch.state_t = batch.state_t_and_tp1[:-1]
        batch.sn = batch.state_t_and_tp1[1:]
        batch.action = U.to_np(batch.action)
        batch.reward = U.to_np(batch.reward)
        batch.done = U.to_np(batch.done)

        batch = self.transform_batch(batch)

        return batch
