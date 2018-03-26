import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class BasicDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data.items()}


class DataGenerator(DataLoader):
    def __init__(self, data, *args, **kwargs):
        dataset = BasicDataset(data)

        super().__init__(
            dataset, *args, collate_fn=custom_collate, drop_last=True, **kwargs)


def custom_collate(batch):
    if torch.is_tensor(batch[0]) or isinstance(batch[0], Variable):
        return torch.stack(batch)

    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch)

    elif isinstance(batch[0], dict):
        return {key: custom_collate([d[key] for d in batch]) for key in batch[0]}

    else:
        return np.stack(batch)
