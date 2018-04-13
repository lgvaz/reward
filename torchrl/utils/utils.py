import numpy as np
from collections import OrderedDict
from scipy.signal import lfilter
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


def to_numpy(tensor):
    if isinstance(tensor, Variable):
        tensor = tensor.data

    return tensor.cpu().numpy()


def explained_var(target, preds):
    return 1 - (target - preds).var() / target.var()


def normalize(array):
    return (array - np.mean(array)) / (np.std(array) + EPSILON)


def discounted_sum_rewards_single(rewards, gamma=0.99):
    # Copy needed because torch doesn't support negative strides
    return lfilter([1.0], [1.0, -gamma], rewards[::-1])[::-1].copy()


def discounted_sum_rewards(rewards, dones=None, gamma=0.99):
    cuts = np.argwhere(dones == 1)
    cuts = np.concatenate((-1 * np.ones((1, 1)), cuts, -1 * np.ones(
        (1, 1)))).squeeze().astype(int)

    returns = [
        discounted_sum_rewards_single(rewards[start + 1:end + 1], gamma=gamma)
        for start, end in zip(cuts[:-1], cuts[1:])
    ]

    return np.concatenate(returns)


def gae_estimation(rewards, state_values, gamma=0.99, gae_lambda=0.95):
    td_target = rewards + gamma * np.append(state_values[1:], 0)
    td_residual = td_target - state_values

    advantages = discounted_sum_rewards(td_residual, gamma * gae_lambda)
    vtarget = advantages + state_values

    return advantages, vtarget
