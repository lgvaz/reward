import numpy as np
import torch
from numbers import Number
from collections import OrderedDict
from torch.autograd import Variable
from torchrl.utils import EPSILON
import cv2
from functools import wraps


def get_obj(config):
    """
    Creates an object based on the given config.

    Parameters
    ----------
    config: dict
        A dict containing the function and the parameters for creating the object.

    Returns
    -------
    obj
        The created object.
    """
    func = config.pop("func")
    obj = func(**config)
    config["func"] = func

    return obj


def env_from_config(config):
    """
    Tries to create an environment from a configuration obj.

    Parameters
    ----------
    config: Config
        Configuration file containing the environment function.

    Returns
    -------
    env: torchrl.envs
        A torchrl environment.

    Raises
    ------
    AttributeError
        If no env is defined in the config obj.
    """
    try:
        env = get_obj(config.env.as_dict())
    except AttributeError:
        raise ValueError(
            "The env must be defined in the config " "or passed as an argument"
        )
    return env


def to_np(value):
    if isinstance(value, Number):
        return np.array(value)
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, LazyArray):
        return np.array(value)

    # If iterable
    try:
        return np.array([to_np(v) for v in value])
    except TypeError:
        return np.array(value)

    raise ValueError("Data type {} not supported".format(value.__class__.__name__))


# TODO: What to do with other types? lists, etc..
def to_tensor(x, cuda_default=True):
    if isinstance(x, np.ndarray):
        # pytorch doesn't support bool
        if x.dtype == "bool":
            x = x.astype("int")
        # we want only single precision floats
        if x.dtype == "float64":
            x = x.astype("float32")

        x = torch.from_numpy(x)

    if isinstance(x, torch.Tensor) and cuda_default and torch.cuda.is_available():
        x = x.cuda()

    return x


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
    return (array - np.mean(array)) / (np.std(array) + EPSILON)


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


class LazyArray:
    """
    Inspired by OpenAI `LazyFrames <https://goo.gl/nTmVW8>`_ this object stores numpy
    arrays as lists, so no unnecessary memory is used when storing arrays that point to
    the same memory, this is a memory optimization trick for the `ReplayBuffer`.

    Beyond this optimization, an optional transform function can be passed, this function
    is executed lazily only when `LazyFrames` gets converted to a numpy array.

    Parameters
    ----------
    data: list
        A list of numpy arrays.
    transform: function
        A function that is applied lazily to the array.
    """

    def __init__(self, data, transform=None, **kwargs):
        self.data = data
        self.transform = transform
        self.kwargs = kwargs

    def __array__(self):
        arr = to_np(self.data, **self.kwargs)
        if self.transform is not None:
            arr = self.transform(arr)
        return arr

    def __iter__(self):
        for v in self.data:
            yield LazyArray(v, **self.kwargs)

    @property
    def shape(self):
        return self.__array__().shape
