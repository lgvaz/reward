import numpy as np
from scipy.signal import lfilter


def td_target(rewards, dones, state_values, gamma=0.99):
    return rewards + (1 - dones) * gamma * np.append(state_values[1:], 0)


# TODO:
def n_step_return():
    pass


def discounted_sum_rewards_single(rewards, gamma=0.99):
    # Copy needed because torch doesn't support negative strides
    return lfilter([1.0], [1.0, -gamma], rewards[::-1])[::-1].copy()


def discounted_sum_rewards(rewards, dones=None, last_state_value=None, gamma=0.99):
    cuts = (np.argwhere(dones == 1).ravel() + 1).tolist()
    cuts.insert(0, 0)
    if not dones[-1]:
        assert last_state_value is not None, \
            'For a incomplete trajectory a value must be passed for last_state_value'
        cuts.append(None)
        rewards = np.append(rewards, last_state_value)

    returns = [
        discounted_sum_rewards_single(rewards[start:end], gamma=gamma)
        for start, end in zip(cuts[:-1], cuts[1:])
    ]

    returns = np.concatenate(returns)

    return returns if dones[-1] else returns[:-1]


def gae_estimation(rewards, dones, state_values, gamma=0.99, gae_lambda=0.95):
    td_target_value = td_target(rewards, dones, state_values, gamma)
    td_residual = td_target_value - state_values

    advantages = discounted_sum_rewards(
        rewards=td_residual,
        dones=dones,
        last_state_value=state_values[-1],
        gamma=gamma * gae_lambda)
    # vtarget = advantages + state_values

    return advantages
