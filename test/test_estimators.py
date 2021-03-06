import pytest
import numpy as np
import reward.utils as U

from reward.utils.estim.estimation_funcs import (
    disc_sum_rs,
    gae_estimation,
    td_target,
)
from .utils import before_after_equal


@pytest.fixture
@before_after_equal
def ds():
    return np.array([[0, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 0, 0]]).T


@pytest.fixture
@before_after_equal
def rs():
    return np.array(
        [
            [0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0, 2.0],
            [1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0],
        ]
    ).T


@pytest.fixture
@before_after_equal
def s_value_t_and_tp1():
    return np.array(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 42.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 42.0],
        ]
    ).T


@pytest.fixture
@before_after_equal
def s_value_t(s_value_t_and_tp1):
    return s_value_t_and_tp1[:-1]


@pytest.fixture
@before_after_equal
def s_value_tp1(s_value_t_and_tp1):
    return s_value_t_and_tp1[1:]


@pytest.fixture
def gamma():
    return 0.9


@pytest.fixture
def gae_lambda():
    return 0.7


@pytest.mark.parametrize("tensor", [False, True])
def test_disc_sum_rs(rs, ds, s_value_t, gamma, tensor):
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

    if tensor:
        rs, ds, s_value_t = map(U.to_tensor, (rs, ds, s_value_t))

    result = disc_sum_rs(rs=rs, ds=ds, vt_last=s_value_t[-1], gamma=gamma)

    np.testing.assert_allclose(U.to_np(result), expected_discounted_return)


@pytest.mark.parametrize("tensor", [False, True])
def test_td_target(rs, ds, s_value_tp1, gamma, tensor):
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

    if tensor:
        rs, ds, s_value_tp1 = map(U.to_tensor, (rs, ds, s_value_tp1))
    result = td_target(rs=rs, ds=ds, v_tp1=s_value_tp1, gamma=gamma)
    np.testing.assert_allclose(U.to_np(result), expected_td_target)


@pytest.mark.parametrize("tensor", [False, True])
def test_gae_estimation(rs, ds, s_value_t, s_value_tp1, gamma, gae_lambda, tensor):
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
    td_residuals = expected_td_target - s_value_t
    expected_gae = np.zeros(rs.shape)
    gae_sum = np.zeros(expected_gae.shape[1])
    for i in reversed(range(rs.shape[0])):
        gae_sum = td_residuals[i] + (1 - ds[i]) * (gamma * gae_lambda) * (gae_sum)
        expected_gae[i] = gae_sum

    if tensor:
        rs, ds, s_value_t, s_value_tp1 = map(
            U.to_tensor, (rs, ds, s_value_t, s_value_tp1)
        )
    result = gae_estimation(
        rs=rs,
        ds=ds,
        v_t=s_value_t,
        v_tp1=s_value_tp1,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    np.testing.assert_allclose(U.to_np(result), expected_gae)
