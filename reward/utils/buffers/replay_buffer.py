import random
from reward.utils import Batch, to_np


class ReplayBuffer:
    def __init__(self, maxlen, num_envs):
        self.maxlen = maxlen // num_envs
        # self.buffer = deque(maxlen=self.maxlen)
        self.buffer = []
        self.num_envs = num_envs
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return list(self.buffer)[key]

    def add_sample(self, **kwargs):
        # If buffer is not full, add a new element
        if len(self.buffer) < self.maxlen:
            self.buffer.append(None)

        # Store new transition at the appropriate index
        self.buffer[self.position] = kwargs
        self.position = (self.position + 1) % self.maxlen

    def sample(self, batch_size):
        envs = random.choices(range(self.num_envs), k=batch_size)
        samples = random.sample(self.buffer, k=batch_size)
        samples = [
            {k: to_np(v)[i] for k, v in sample.items()}
            for i, sample in zip(envs, samples)
        ]
        batch = Batch.from_list_of_dicts(samples)
        batch = batch.apply_to_all(to_np)
        batch = batch.apply_to_all(lambda x: x[None])

        return batch
