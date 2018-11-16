import torch
import numpy as np
import reward.utils as U
from reward.batcher import BaseBatcher


class RolloutBatcher(BaseBatcher):
    def get_batch(self, act_fn):
        if self.s is None:
            self.s = self.transform_state(self.runner.reset())
            self.s = U.to_tensor(self.s)

        horizon = self.batch_size // self.runner.num_envs
        batch = U.Batch(initial_keys=["s_and_tp1", "action", "reward", "done"])

        for i in range(horizon):
            action = act_fn(self.s, self.num_steps)

            sn, reward, done, info = self.runner.act(action)
            sn = U.to_tensor(self.transform_state(sn))

            batch.s_and_tp1.append(self.s)
            batch.action.append(action)
            batch.reward.append(reward)
            batch.done.append(done)
            # batch.info.append(info)

            self.s = sn
        batch.s_and_tp1.append(self.s)

        batch.s_and_tp1 = torch.stack(batch.s_and_tp1)
        batch.s = batch.s_and_tp1[:-1]
        batch.sn = batch.s_and_tp1[1:]
        batch.action = U.to_np(batch.action)
        batch.reward = U.to_np(batch.reward)
        batch.done = U.to_np(batch.done)

        batch = self.transform_batch(batch)

        return batch
