import numpy as np
from reward.utils import Batch, to_np


class ReplayBuffer:
    def __init__(self, maxlen, num_envs):
        self.maxlen = maxlen
        self.num_envs = num_envs
        self.buffer = []
        # Intialized at -1 so the first updated position is 0
        self.position = -1

    def __len__(self):
        return len(self.buffer) * self.num_envs

    def __getitem__(self, key):
        row, col = key // self.num_envs, key % self.num_envs
        return {k: to_np(v)[col] for k, v in self.buffer[row].items()}

    def _get_batch(self, idxs):
        samples = [self[i] for i in idxs]
        batch = Batch.from_list_of_dicts(samples)
        batch = batch.apply_to_all(to_np)
        batch = batch.apply_to_all(lambda x: x[None])
        batch.idx = idxs
        return batch

    def add_sample(self, **kwargs):
        # If buffer is not full, add a new element
        if len(self.buffer) + self.num_envs <= self.maxlen:
            self.buffer.append(None)
        # Store new transition at the appropriate index
        self.position = (self.position + 1) % (self.maxlen // self.num_envs)
        self.buffer[self.position] = kwargs

    def sample(self, batch_size):
        idxs = np.random.choice(len(self), batch_size, replace=False)
        return self._get_batch(idxs=idxs)
