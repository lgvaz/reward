import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from reward.utils.memories import SimpleMemory
from reward.utils import to_tensor, join_first_dims, maybe_np, to_np


class Batch(SimpleMemory):
    def __len__(self):
        return len(self["state_t"])

    # TODO: apply inplace instead of returning new batch
    def apply_to_all(self, func):
        return Batch((k, func(v)) for k, v in self.items())

    def apply_to_keys(self, func, keys):
        return Batch((k, func(self[k])) for k in keys)

    def sample(self, mini_batches, shuffle):
        keys = list(self.keys())

        return self.sample_keys(keys=keys, mini_batches=mini_batches, shuffle=shuffle)

    def sample_keys(self, keys, mini_batches, shuffle):
        self["idxs"] = torch.arange(len(self)).long()
        keys = keys + ["idxs"]

        if mini_batches > 1:
            values = [self[k] for k in keys]
            batch_size = len(self) // mini_batches

            dataset = TensorDataset(*values)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

            for data in data_loader:
                yield Batch((k, v) for k, v in zip(keys, data))
        else:
            yield self

    def concat_batch(self):
        # func = lambda x: x.reshape((-1, *x.shape[2:])) if (
        #     isinstance(x, (np.ndarray, torch.Tensor))) else x
        func = (
            lambda x: join_first_dims(x, num_dims=2)
            if (isinstance(x, (np.ndarray, torch.Tensor)))
            else x
        )
        return self.apply_to_all(func)

    def to_tensor(self):
        return Batch({k: to_tensor(v) for k, v in self.items()})

    @classmethod
    def from_trajs(cls, trajs):
        return cls(**Batch.concat_trajectories(trajs))

    @staticmethod
    def concat_trajectories(trajs):
        batch = dict()
        for key in trajs[0]:
            batch[key] = np.concatenate([t[key] for t in trajs])

        return batch
