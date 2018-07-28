import pytest

import numpy as np

from torchrl.utils.estimators.estimation_funcs import (
    discounted_sum_rewards,
    gae_estimation,
    td_target,
)
from .utils import before_after_equal


@pytest.fixture
@before_after_equal
def dones():
    return np.array([[0, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 0, 0]]).T


@pytest.fixture
@before_after_equal
def rewards():
    return np.array(
        [
            [0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0, 2.0],
            [1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0],
        ]
    ).T


@pytest.fixture
@before_after_equal
def state_value_t_and_tp1():
    return np.array(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 42.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 42.0],
        ]
    ).T


@pytest.fixture
@before_after_equal
def state_value_t(state_value_t_and_tp1):
    return state_value_t_and_tp1[:-1]


@pytest.fixture
@before_after_equal
def state_value_tp1(state_value_t_and_tp1):
    return state_value_t_and_tp1[1:]


@pytest.fixture
def gamma():
    return 0.9


@pytest.fixture
def gae_lambda():
    return 0.7


def test_discounted_sum_rewards(rewards, dones, state_value_t, gamma):
    expected_discounted_return = np.array(
        [
            [
                gamma * (1.0 + 3.0 * gamma),
                1.0 + 3.0 * gamma,
                3.0,
                0.0,
                0.0,
                gamma * (1.0 + gamma * 2.0),
                1.0 + gamma * 2.0,
                2.0,
            ],
            [1.0, 0.0, 0.0, 0.0, 2.0, 1.0 + gamma * (gamma * 5.0), gamma * 5.0, 5.0],
        ]
    ).T

    result = discounted_sum_rewards(
        rewards=rewards, dones=dones, last_state_value_t=state_value_t[-1], gamma=gamma
    )

    np.testing.assert_equal(result, expected_discounted_return)


def test_td_target(rewards, dones, state_value_tp1, gamma):
    expected_td_target = np.array(
        [
            [
                0.0 + gamma * 5.0,
                1.0 + gamma * 5.0,
                3.0 + gamma * 5.0,
                0.0 + gamma * 5.0,
                0,
                0.0 + gamma * 5.0,
                1.0 + gamma * 5.0,
                2,
            ],
            [
                1,
                0.0 + gamma * 5.0,
                0.0 + gamma * 5.0,
                0,
                2,
                1 + gamma * 5.0,
                0.0 + gamma * 5.0,
                1 + gamma * 42.0,
            ],
        ]
    ).T

    result = td_target(
        rewards=rewards, dones=dones, state_value_tp1=state_value_tp1, gamma=gamma
    )

    np.testing.assert_equal(result, expected_td_target)


def test_gae_estimation(
    rewards, dones, state_value_t, state_value_t_and_tp1, gamma, gae_lambda
):
    expected_td_target = np.array(
        [
            [
                0.0 + gamma * 5.0,
                1.0 + gamma * 5.0,
                3.0 + gamma * 5.0,
                0.0 + gamma * 5.0,
                0,
                0.0 + gamma * 5.0,
                1.0 + gamma * 5.0,
                2,
            ],
            [
                1,
                0.0 + gamma * 5.0,
                0.0 + gamma * 5.0,
                0,
                2,
                1 + gamma * 5.0,
                0.0 + gamma * 5.0,
                1 + gamma * 42.0,
            ],
        ]
    ).T
    td_residuals = expected_td_target - state_value_t
    expected_gae = np.zeros(rewards.shape)
    gae_sum = np.zeros(expected_gae.shape[1])
    for i in reversed(range(rewards.shape[0])):
        gae_sum = td_residuals[i] + (1 - dones[i]) * (gamma * gae_lambda) * (gae_sum)
        expected_gae[i] = gae_sum

    result = gae_estimation(
        rewards=rewards,
        dones=dones,
        state_value_t_and_tp1=state_value_t_and_tp1,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    np.testing.assert_equal(result, expected_gae)
