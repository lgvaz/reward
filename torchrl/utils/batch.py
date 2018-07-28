import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from torchrl.utils.memories import SimpleMemory
from torchrl.utils import to_tensor, join_first_dims


class Batch(SimpleMemory):
    def __len__(self):
        return len(self["state_t"])

    def apply_to_all(self, func):
        return Batch((k, func(v)) for k, v in self.items())

    def apply_to_keys(self, func, keys):
        return Batch((k, func(self[k])) for k in keys)

    def sample(self, num_mini_batches, shuffle):
        keys = list(self.keys())

        return self.sample_keys(
            keys=keys, num_mini_batches=num_mini_batches, shuffle=shuffle
        )

    def sample_keys(self, keys, num_mini_batches, shuffle):
        self["idxs"] = torch.arange(len(self)).long()
        keys = keys + ["idxs"]

        if num_mini_batches > 1:
            values = [self[k] for k in keys]
            batch_size = len(self) // num_mini_batches

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

    def to_array_or_tensor(self):
        new_batch = Batch()
        for k, v in self.items():
            if isinstance(v[0], np.ndarray):
                new_batch[k] = np.stack(v)

            elif isinstance(v[0], torch.Tensor):
                new_batch[k] = torch.stack(v)

            else:
                new_batch[k] = v

        return new_batch

    @classmethod
    def from_trajs(cls, trajs):
        return cls(**Batch.concat_trajectories(trajs))

    @staticmethod
    def concat_trajectories(trajs):
        batch = dict()
        for key in trajs[0]:
            batch[key] = np.concatenate([t[key] for t in trajs])

        return batch
