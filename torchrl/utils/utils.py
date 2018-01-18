import numpy as np
from collections import OrderedDict
from scipy.signal import lfilter
from torchrl.utils import EPSILON


def get_obj(config):
    func = config.pop('func')
    obj = func(**config)
    config['func'] = func

    return obj


def normalize(array):
    return (array - np.mean(array)) / (np.std(array) + EPSILON)


def discounted_sum_rewards(rewards, gamma=0.99):
    # Copy needed because torch doesn't support negative strides
    return lfilter([1.0], [1.0, -gamma], rewards[::-1])[::-1].copy()


def gae_estimation(rewards, state_values, gamma=0.99, gae_lambda=0.95):
    td_target = rewards + gamma * np.append(state_values[1:], 0)
    td_residual = td_target - state_values

    advantages = discounted_sum_rewards(td_residual, gamma * gae_lambda)
    vtarget = advantages + state_values

    return advantages, vtarget
