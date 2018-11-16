import numpy as np
import torch
from numbers import Number
from reward.utils import EPSILON


def to_np(v):
    if isinstance(v, Number): return np.array(v)
    if isinstance(v, (np.ndarray, np.generic)): return v
    if isinstance(v, torch.Tensor): return v.detach().cpu().numpy()
    # If iterable
    try: return np.array([to_np(v) for v in v])
    except TypeError: return np.array(v)

    raise ValueError("Data type {} not supported".format(v.__class__.__name__))


def explained_var(target, preds):
    """
    Calculates the explained variance between two datasets.
    Useful for estimating the quality of the value function

    Parameters
    ----------
    target: np.array
        Target dataset.
    preds: np.array
        Predictions array.

    Returns
    -------
    float
        The explained variance.
    """
    return 1 - (target.squeeze() - preds.squeeze()).var() / target.view(-1).var()


def normalize(array):
    """
    Normalize an array by subtracting the mean and diving by the std dev.
    """
    # return (array - np.mean(array)) / (np.std(array) + EPSILON)
    return (array - array.mean()) / (array.std() + EPSILON)


def map_range(array, low, high):
    norm_array = (array - array.min()) / (array.max() - array.min())
    return low + (norm_array * (high - low))


def one_hot(array, num_classes):
    return np.eye(num_classes)[array]


def make_callable(x):
    if callable(x):
        return x
    try:
        return [make_callable(v) for v in x]
    except TypeError:
        return lambda *args, **kwargs: x


def join_first_dims(x, num_dims):
    return x.reshape((-1, *x.shape[num_dims:]))
