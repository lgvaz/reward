from collections import OrderedDict

from scipy.signal import lfilter


def get_obj(config):
    func = config.pop('func')
    obj = func(**config)
    config['func'] = func

    return obj


def discounted_sum_rewards(rewards, gamma=0.99):
    # Copy needed because torch doesn't support negative strides
    return lfilter([1.0], [1.0, -gamma], rewards[::-1])[::-1].copy()
