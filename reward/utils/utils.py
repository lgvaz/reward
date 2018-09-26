import numpy as np
import torch
from numbers import Number
from collections import OrderedDict
from torch.autograd import Variable
from reward.utils import EPSILON
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
    env: reward.envs
        A reward environment.

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

    # If iterable
    try:
        return np.array([to_np(v) for v in value])
    except TypeError:
        return np.array(value)

    raise ValueError("Data type {} not supported".format(value.__class__.__name__))


# TODO: Depecrated
def maybe_np(value):
    if isinstance(value, torch.Tensor):
        return value
    else:
        try:
            return to_np(value)
        except ValueError:
            return value


# TODO: What to do with other types? lists, etc..
def to_tensor(x, cuda_default=True):
    if not isinstance(x, torch.Tensor):
        x = to_np(x)

    if isinstance(x, np.ndarray):
        # TODO: Everything to float??
        # pytorch doesn't support bool
        if x.dtype == "bool":
            x = x.astype("float32")
        # we want only single precision floats
        if x.dtype == "float64":
            x = x.astype("float32")
        # TODO: this may break something
        if x.dtype == "int":
            x = x.astype("float32")

        x = torch.from_numpy(x)

    if isinstance(x, torch.Tensor) and cuda_default and torch.cuda.is_available():
        x = x.cuda()

    else:
        raise ValueError("{} not suported".format(type(x)))

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
    # return (array - np.mean(array)) / (np.std(array) + EPSILON)
    return (array - array.mean()) / (array.std() + EPSILON)


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
