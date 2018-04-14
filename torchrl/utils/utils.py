import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from torchrl.utils import EPSILON


def get_obj(config):
    func = config.pop('func')
    obj = func(**config)
    config['func'] = func

    return obj


def env_from_config(config):
    try:
        env = get_obj(config.env.obj)
    except AttributeError:
        raise ValueError('The env must be defined in the config '
                         'or passed as an argument')
    return env


# TODO: Variable deprecated
def to_numpy(tensor):
    # if isinstance(tensor, Variable):
    #     tensor = tensor.data

    return tensor.detach().cpu().numpy()


def explained_var(target, preds):
    return 1 - (target - preds).var() / target.var()


def normalize(array):
    return (array - np.mean(array)) / (np.std(array) + EPSILON)
