import numpy as np


def discounted_sum_rewards(rewards, dones, last_state_value_t=None, gamma=0.99):
    """
    Expected shape: (num_samples, num_envs)
    """
    rewards = rewards.copy()

    # if last_state_value_t is None and not all(dones[-1]):
    #     raise AssertionError('If one episode is not finished you must'
    #                          'pass a value to last_state_value_t to bootstrap from.')

    if last_state_value_t is not None:
        bootstrap = np.where(dones[-1] == False)[0]
        rewards[-1][bootstrap] = last_state_value_t[bootstrap]

    returns = np.zeros(rewards.shape)
    returns_sum = np.zeros(rewards.shape[-1])

    for i in reversed(range(rewards.shape[0])):
        returns_sum = rewards[i] + gamma * returns_sum * (1 - dones[i])
        returns[i] = returns_sum

    return returns


def td_target(*, rewards, dones, state_value_tp1, gamma):
    return rewards + (1 - dones) * gamma * state_value_tp1


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
