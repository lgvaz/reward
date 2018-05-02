import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from torchrl.utils import EPSILON, SimpleMemory


def get_obj(config):
    func = config.pop('func')
    obj = func(**config)
    config['func'] = func

    return obj


def env_from_config(config):
    try:
        env = get_obj(config.env.as_dict())
    except AttributeError:
        raise ValueError('The env must be defined in the config '
                         'or passed as an argument')
    return env


def join_transitions(transitions):
    trajectory = SimpleMemory(
        (key, np.array([t[key] for t in transitions])) for key in transitions[0])

    return trajectory


# TODO: Variable deprecated
def to_numpy(tensor):
    # if isinstance(tensor, Variable):
    #     tensor = tensor.data

    return tensor.detach().cpu().numpy()


def explained_var(target, preds):
    return 1 - (target.view(-1) - preds.view(-1)).var() / target.view(-1).var()


def normalize(array):
    return (array - np.mean(array)) / (np.std(array) + EPSILON)
