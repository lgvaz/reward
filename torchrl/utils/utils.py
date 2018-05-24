import numpy as np
from numbers import Number
from collections import OrderedDict
from torch.autograd import Variable
from torchrl.utils import EPSILON, SimpleMemory


def get_obj(config):
    '''
    Creates an object based on the given config.

    Parameters
    ----------
    config: dict
        A dict containing the function and the parameters for creating the object.

    Returns
    -------
    obj
        The created object.
    '''
    func = config.pop('func')
    obj = func(**config)
    config['func'] = func

    return obj


def env_from_config(config):
    '''
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
    '''
    try:
        env = get_obj(config.env.as_dict())
    except AttributeError:
        raise ValueError('The env must be defined in the config '
                         'or passed as an argument')
    return env


def join_transitions(transitions):
    '''
    Joins a list of transitions into a single trajectory.
    '''
    trajectory = SimpleMemory(
        (key, np.array([t[key] for t in transitions])) for key in transitions[0])

    return trajectory


def to_np(value):
    '''
    Convert a value to a numpy array.
    '''
    if isinstance(value, Number):
        return np.array(value)
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.array(value)

    return value.detach().cpu().numpy()


def explained_var(target, preds):
    '''
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
    '''
    return 1 - (target.view(-1) - preds.view(-1)).var() / target.view(-1).var()


def normalize(array):
    '''
    Normalize an array by subtracting the mean and diving by the std dev.
    '''
    return (array - np.mean(array)) / (np.std(array) + EPSILON)


def one_hot(array, num_classes):
    return np.eye(num_classes)[array]


def make_callable(x):
    if callable(x):
        return x
    else:
        return lambda *args, **kwargs: x
