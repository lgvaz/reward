import torch
import numpy as np
from reward.utils import to_tensor
from copy import deepcopy


def discounted_sum_rewards(rewards, dones, last_state_value_t=None, gamma=0.99):
    """
    Expected shape: (num_samples, num_envs)
    """
    rewards = deepcopy(rewards)

    # TODO: This works but is messy
    is_tensor = lambda x: isinstance(x, torch.Tensor)
    if any(map(is_tensor, (rewards, dones, last_state_value_t))):
        rewards, dones, last_state_value_t = [
            to_tensor(x) for x in [rewards, dones, last_state_value_t]
        ]
        tensor = True
    else:
        tensor = False

    # if last_state_value_t is None and not all(dones[-1]):
    #     raise AssertionError('If one episode is not finished you must'
    #                          'pass a value to last_state_value_t to bootstrap from.')

    if last_state_value_t is not None:
        rewards[-1][dones[-1] == 0] = last_state_value_t[dones[-1] == 0]

    returns = np.zeros(rewards.shape, dtype="float32")
    returns_sum = np.zeros(rewards.shape[-1], dtype="float32")
    if tensor:
        returns, returns_sum = map(to_tensor, (returns, returns_sum))

    for i in reversed(range(rewards.shape[0])):
        returns_sum = rewards[i] + gamma * returns_sum * (1 - dones[i])
        returns[i] = returns_sum

    return returns


def td_target(*, rewards, dones, v_tp1, gamma):
    return rewards + (1 - dones) * gamma * v_tp1


# TODO: Test this
def q_learning_target(*, rewards, dones, q_tp1, gamma):
    max_q_tp1, _ = q_tp1.max(dim=1)
    return td_target(
        rewards=rewards, dones=dones, state_value_tp1=max_q_tp1, gamma=gamma
    )


def gae_estimation(
    rewards, dones, state_value_t, state_value_tp1, *, gamma, gae_lambda
):
    td_target_value = td_target(
        rewards=rewards, dones=dones, state_value_tp1=state_value_tp1, gamma=gamma
    )
    td_residual = td_target_value - state_value_t

    advantages = discounted_sum_rewards(
        rewards=td_residual,
        dones=dones,
        last_state_value_t=None,
        gamma=gamma * gae_lambda,
    )

    return advantages
