import numpy as np
import torch
from numbers import Number
from collections import OrderedDict
from torch.autograd import Variable
from torchrl.utils import EPSILON, SimpleMemory
import cv2
from functools import wraps


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


def to_tensor(x, cuda_default=True):
    if isinstance(x, np.ndarray):
        # pytorch doesn't support bool
        if x.dtype == 'bool':
            x = x.astype('int')
        # we want only single precision floats
        if x.dtype == 'float64':
            x = x.astype('float32')

        x = torch.from_numpy(x)

    if isinstance(x, torch.Tensor) and cuda_default and torch.cuda.is_available():
        x = x.cuda()

    return x


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
    return 1 - (target.squeeze() - preds.squeeze()).var() / target.view(-1).var()


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


# def rgb_to_gray():
#     @wraps(rgb_to_gray)
#     def get(frame):
#         return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[..., None]

#     return get

# def rescale_img(shape):
#     @wraps(rescale_img)
#     def get(frame):
#         assert frame.ndim == 3 or frame.ndim == 2
#         frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)

#         return frame if frame.ndim == 3 else frame[:, :, None]

#     return get

# # def force_shape(ndim=4):
# #     def get(frame):
# #         assert frame.ndim >= 2, \
# #             'frame have {} dimensions and should have at least 2'.format(frame.ndim)
# #         for _ in range(ndim - frame.ndim):
# #             frame = frame[None]
# #         return frame

# #     return get

# def hwc_to_chw():
#     @wraps(hwc_to_chw)
#     def get(frame):
#         assert frame.ndim == 3, 'frame have {} dims but must have 3'.format(frame.ndim)
#         return np.rollaxis(frame, -1)

#     return get
