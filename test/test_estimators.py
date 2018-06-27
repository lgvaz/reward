import unittest

import numpy as np

from torchrl.utils.estimators.estimation_funcs import (discounted_sum_rewards,
                                                       gae_estimation, td_target)
from .timer import timeit


class BaseEstimationTest(unittest.TestCase):
    def setUp(self):
        self.gamma = 0.9
        self.gae_lambda = 0.7

        self.rewards = np.array([[0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0, 2.0],
                                 [1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0]]).T

        self.dones = np.array([[0, 0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 0, 0]]).T

        self.state_value_t_and_tp1 = np.array(
            [[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 42.0],
             [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 42.0]]).T
        self.state_value_t = self.state_value_t_and_tp1[:-1]
        self.state_value_tp1 = self.state_value_t_and_tp1[1:]

        self._rewards = self.rewards.copy()
        self._dones = self.dones.copy()
        self._state_value_t_and_tp1 = self.state_value_t_and_tp1.copy()
        self._state_value_t = self.state_value_t.copy()
        self._state_value_tp1 = self.state_value_tp1.copy()

    def tearDown(self):
        msg = 'This array should not be modified'
        # Check that no values have been altered
        self.assertTrue((self.rewards == self._rewards).all(), msg)
        self.assertTrue((self.dones == self._dones).all(), msg)
        self.assertTrue((self.state_value_t_and_tp1 == self._state_value_t_and_tp1).all(),
                        msg)
        self.assertTrue((self.state_value_t == self._state_value_t).all(), msg)
        self.assertTrue((self.state_value_tp1 == self._state_value_tp1).all(), msg)


class TestEstimationFuncs(BaseEstimationTest):
    @timeit
    def test_discounted_sum_rewards(self):
        expected_discounted_return = np.array([[
            self.gamma * (1.0 + 3.0 * self.gamma), 1.0 + 3.0 * self.gamma, 3.0, 0.0, 0.0,
            self.gamma * (1.0 + self.gamma * 2.0), 1.0 + self.gamma * 2.0, 2.0
        ], [
            1.0, 0.0, 0.0, 0.0, 2.0, 1.0 + self.gamma * (self.gamma * 5.0),
            self.gamma * 5.0, 5.0
        ]]).T

        result = discounted_sum_rewards(
            rewards=self.rewards,
            dones=self.dones,
            last_state_value_t=self.state_value_t[-1],
            gamma=self.gamma)

        self.assertTrue((result == expected_discounted_return).all())

    @timeit
    def test_td_target(self):
        expected_td_target = np.array([[
            0.0 + self.gamma * 5.0, 1.0 + self.gamma * 5.0, 3.0 + self.gamma * 5.0,
            0.0 + self.gamma * 5.0, 0, 0.0 + self.gamma * 5.0, 1.0 + self.gamma * 5.0, 2
        ], [
            1, 0.0 + self.gamma * 5.0, 0.0 + self.gamma * 5.0, 0, 2, 1 + self.gamma * 5.0,
            0.0 + self.gamma * 5.0, 1 + self.gamma * 42.0
        ]]).T

        result = td_target(
            rewards=self.rewards,
            dones=self.dones,
            state_value_tp1=self.state_value_tp1,
            gamma=self.gamma)

        self.assertTrue((result == expected_td_target).all())

    @timeit
    def test_gae_estimation(self):
        expected_td_target = np.array([[
            0.0 + self.gamma * 5.0, 1.0 + self.gamma * 5.0, 3.0 + self.gamma * 5.0,
            0.0 + self.gamma * 5.0, 0, 0.0 + self.gamma * 5.0, 1.0 + self.gamma * 5.0, 2
        ], [
            1, 0.0 + self.gamma * 5.0, 0.0 + self.gamma * 5.0, 0, 2, 1 + self.gamma * 5.0,
            0.0 + self.gamma * 5.0, 1 + self.gamma * 42.0
        ]]).T
        td_residuals = (expected_td_target - self.state_value_t)
        expected_gae = np.zeros(self.rewards.shape)
        gae_sum = np.zeros(expected_gae.shape[1])
        for i in reversed(range(self.rewards.shape[0])):
            gae_sum = td_residuals[i] + (1 - self.dones[i]) * (
                self.gamma * self.gae_lambda) * (gae_sum)
            expected_gae[i] = gae_sum

        result = gae_estimation(
            rewards=self.rewards,
            dones=self.dones,
            state_value_t_and_tp1=self.state_value_t_and_tp1,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda)

        self.assertTrue((result == expected_gae).all())
