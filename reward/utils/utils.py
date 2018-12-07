import numpy as np
import torch
from numbers import Number
from reward.utils import EPSILON
from collections import Iterable


def to_np(v): return v.detach().cpu().numpy()

def is_np(v): return isinstance(v, (np.ndarray, np.generic))

def listify(p=None):
    if p is None:                   return []
    elif not isinstance(p, list): return [p]
    else:                           return p

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
    "Normalize an array by subtracting the mean and diving by the std dev."
    return (array - array.mean()) / (array.std() + EPSILON)

def map_range(old_low, old_high, new_low, new_high):
    old_span = old_high - old_low
    new_span = new_high - new_low
    def get(value):
        norm_value = (value - old_low) / old_span
        return new_low + (norm_value * new_span)
    return get

def make_callable(x):
    if callable(x): return x
    try:              return [make_callable(v) for v in x]
    except TypeError: return lambda *args, **kwargs: x

def one_hot(array, num_classes): return np.eye(num_classes)[array]

def join_first_dims(x, num_dims): return x.reshape((-1, *x.shape[num_dims:]))
