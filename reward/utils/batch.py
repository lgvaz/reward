import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from reward.utils.memories import SimpleMemory
from reward.utils import to_tensor, join_first_dims, to_np


class Batch(SimpleMemory):
    def __len__(self):
        return len(self["state_t"])

    def apply_to_all(self, func):
        return Batch((k, func(v)) for k, v in self.items())

    def apply_to_keys(self, func, keys):
        return Batch((k, func(self[k])) for k in keys)

    def concat_batch(self):
        func = (
            lambda x: join_first_dims(x, num_dims=2)
            if (isinstance(x, (np.ndarray, torch.Tensor)))
            else x
        )
        return self.apply_to_all(func)

    def to_tensor(self, ignore=["idx"]):
        return Batch({k: v if k in ignore else to_tensor(v) for k, v in self.items()})
