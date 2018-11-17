import torch
import numpy as np
import reward.utils as U
from reward.batcher import BaseBatcher


class RolloutBatcher(BaseBatcher):
    def get_batch(self, act_fn):
        if self.s is None:
            self.s = self.transform_s(self.runner.reset())
            self.s = U.to_tensor(self.s)

        horizon = self.batch_size // self.runner.num_envs
        batch = U.Batch(initial_keys=["s_and_tp1", "ac", "r", "d"])

        for i in range(horizon):
            ac = act_fn(self.s, self.num_steps)

            sn, r, d, info = self.runner.act(ac)
            sn = U.to_tensor(self.transform_s(sn))

            batch.s_and_tp1.append(self.s)
            batch.ac.append(ac)
            batch.r.append(r)
            batch.d.append(d)
            # batch.info.append(info)

            self.s = sn
        batch.s_and_tp1.append(self.s)

        batch.s_and_tp1 = torch.stack(batch.s_and_tp1)
        batch.s = batch.s_and_tp1[:-1]
        batch.sn = batch.s_and_tp1[1:]
        batch.ac = U.to_np(batch.ac)
        batch.r = U.to_np(batch.r)
        batch.d = U.to_np(batch.d)

        batch = self.transform_batch(batch)

        return batch
