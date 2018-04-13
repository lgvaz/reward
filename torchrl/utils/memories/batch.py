import numpy as np
from torch.utils.data import Dataset


class Batch(Dataset):
    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return len(self.batch['state_ts'])

    def __setattr__(self, name, value):
        if name == 'batch':
            super().__setattr__(name, value)
        else:
            self.batch[name] = value

    def __getattr__(self, value):
        return self.batch[value]

    def __getitem__(self, i):
        return Batch({k: v[i] for k, v in self.batch.items()})

    def __iter__(self):
        yield from self.batch

    def keys(self):
        yield from self.batch.keys()

    def values(self):
        yield from self.batch.values()

    def items(self):
        yield from self.batch.items()

    def apply_to_all(self, func):
        return Batch({k: func(v) for k, v in self.batch.items()})

    @classmethod
    def from_trajs(cls, trajs):
        return cls(Batch.concat_trajectories(trajs))

    @staticmethod
    def concat_trajectories(trajs):
        batch = dict()
        for key in trajs[0]:
            batch[key] = np.concatenate([t[key] for t in trajs])

        return batch
