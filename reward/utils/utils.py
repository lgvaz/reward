import numpy as np
import torch
from numbers import Number
from reward.utils import EPSILON
from collections import Iterable


def to_np(v): return v.detach().cpu().numpy()

def is_np(v): return isinstance(v, (np.ndarray, np.generic))

def listify(p=None, q=None):
    "Make `p` same length as `q`. From fastai"
    if p is None: p=[]
    elif isinstance(p, str):          p=[p]
    elif not isinstance(p, Iterable): p=[p]
    n = q if type(q)==int else len(p) if q is None else len(q)
    if len(p)==1: p = p * n
    assert len(p)==n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)

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

def map_range(array, low, high):
    norm_array = (array - array.min()) / (array.max() - array.min())
    return low + (norm_array * (high - low))

def make_callable(x):
    if callable(x): return x
    try:              return [make_callable(v) for v in x]
    except TypeError: return lambda *args, **kwargs: x

def one_hot(array, num_classes): return np.eye(num_classes)[array]

def join_first_dims(x, num_dims): return x.reshape((-1, *x.shape[num_dims:]))
