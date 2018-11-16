import torch
import numpy as np
from reward.utils import to_tensor
from copy import deepcopy


def discounted_sum_rewards(rewards, dones, v_t_last=None, gamma=0.99):
    """
    Expected shape: (num_samples, num_envs)
    """
    rewards = deepcopy(rewards)

    # TODO: This works but is messy
    is_tensor = lambda x: isinstance(x, torch.Tensor)
    if any(map(is_tensor, (rewards, dones, v_t_last))):
        rewards, dones = to_tensor(rewards), to_tensor(dones)
        v_last = to_tensor(v_t_last) if v_t_last is not None else v_t_last
        tensor = True
    else: tensor = False

    # if v_t_last is None and not all(dones[-1]):
    #     raise AssertionError('If one episode is not finished you must'
    #                          'pass a value to v_t_last to bootstrap from.')

    if v_t_last is not None: rewards[-1][dones[-1] == 0] = v_t_last[dones[-1] == 0]

    returns = np.zeros(rewards.shape, dtype="float32")
    returns_sum = np.zeros(rewards.shape[-1], dtype="float32")
    if tensor: returns, returns_sum = map(to_tensor, (returns, returns_sum))

    for i in reversed(range(rewards.shape[0])):
        returns_sum = rewards[i] + gamma * returns_sum * (1 - dones[i])
        returns[i] = returns_sum

    return returns

def td_target(*, rewards, dones, v_tp1, gamma):
    return rewards + (1 - dones) * gamma * v_tp1

# TODO: Test this
def q_learning_target(*, rewards, dones, q_tp1, gamma):
    max_q_tp1, _ = q_tp1.max(dim=1)
    return td_target(rewards=rewards, dones=dones, v_tp1=max_q_tp1, gamma=gamma)

def gae_estimation(rewards, dones, v_t, v_tp1, *, gamma, gae_lambda):
    td_target_value = td_target(rewards=rewards, dones=dones, v_tp1=v_tp1, gamma=gamma)
    td_residual = td_target_value - v_t

    advantages = discounted_sum_rewards(
        rewards=td_residual, dones=dones, v_t_last=None, gamma=gamma * gae_lambda
    )

    return advantages
