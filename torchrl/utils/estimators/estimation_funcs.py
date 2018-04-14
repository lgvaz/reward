import numpy as np
from scipy.signal import lfilter


def td_target(rewards, dones, state_values, gamma=0.99):
    return rewards + (1 - dones) * gamma * np.append(state_values[1:], 0)


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


def gae_estimation(rewards, dones, state_values, gamma=0.99, gae_lambda=0.95):
    td_target_value = td_target(rewards, dones, state_values, gamma)
    td_residual = td_target_value - state_values

    advantages = discounted_sum_rewards(td_residual, dones, gamma * gae_lambda)
    # vtarget = advantages + state_values

    return advantages
