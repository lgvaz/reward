import pytest
import unittest
import numpy as np

from torchrl.envs import GymEnv, AtariEnv
from torchrl.envs.wrappers import RGB2GRAY, Rescale
from torchrl.runners import SingleRunner, PAACRunner
from torchrl.batchers import RolloutBatcher

from .timer import timeit

pytestmark = pytest.mark.skip("Need to develop new tests for batchers")


class TestBatchers(unittest.TestCase):
    def setUp(self):
        self.seed = np.random.choice(4200)
        self.batch_size = 200

    def create_runner_trajs(self, runner, actions):
        states, rewards, dones, infos = [], [], [], []

        state = runner.reset()
        for a in actions:
            next_state, reward, done, info = runner.act(a)

            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

            state = next_state

        return list(map(np.array, [states, rewards, dones, infos]))


class TestRolloutBatcher(TestBatchers):
    @timeit
    def test_rollout_batcher_simple(self):
        self.num_steps = 1000

        self._test_rollout_batcher_simple(GymEnv("CartPole-v1"))
        self._test_rollout_batcher_simple(GymEnv("Pendulum-v0"))
        self._test_rollout_batcher_simple(
            Rescale(RGB2GRAY(AtariEnv("PongNoFrameskip-v4")), shape=(84, 84))
        )

    @timeit
    def test_rollout_batcher_paac(self):
        self.num_envs = 16
        self.num_steps = self.num_envs * 5

        self._test_rollout_batcher_paac(
            [GymEnv("CartPole-v1") for _ in range(self.num_envs)]
        )
        self._test_rollout_batcher_paac(
            [GymEnv("Pendulum-v0") for _ in range(self.num_envs)]
        )
        self._test_rollout_batcher_paac(
            [
                Rescale(RGB2GRAY(AtariEnv("PongNoFrameskip-v4")), shape=(84, 84))
                for _ in range(self.num_envs)
            ]
        )

    def _test_rollout_batcher_simple(self, env):
        assert self.num_steps % self.batch_size == 0

        actions = np.array([env.sample_random_action() for _ in range(self.num_steps)])

        # Get expected batch
        env.seed(self.seed)
        runner = SingleRunner(env)
        exp_s, exp_r, exp_d, exp_i = self.create_runner_trajs(runner, actions)

        # Get actual batch
        env.seed(self.seed)
        action_gen = (a for a in actions)
        action_fn = lambda state, step: next(action_gen)

        batcher = RolloutBatcher(runner, batch_size=self.batch_size)

        for i in range(0, self.num_steps, self.batch_size):
            batch = batcher.get_batch(select_action_fn=action_fn)

            np.testing.assert_allclose(batch.state_t, exp_s[i : i + self.batch_size])
            np.testing.assert_allclose(batch.reward, exp_r[i : i + self.batch_size])
            np.testing.assert_allclose(batch.done, exp_d[i : i + self.batch_size])
            np.testing.assert_equal(batch.info, exp_i[i : i + self.batch_size])

        runner.close()
        batcher.close()

    def _test_rollout_batcher_paac(self, envs):
        seeds = np.random.choice(4200, self.num_envs)
        horizon = self.batch_size // self.num_envs
        actions = np.array(
            [
                [envs[0].sample_random_action() for _ in range(self.num_steps)]
                for _ in range(self.num_envs)
            ]
        ).swapaxes(0, 1)

        # Expected
        [env.seed(int(seed)) for env, seed in zip(envs, seeds)]
        runner = PAACRunner(envs)
        exp_s, exp_r, exp_d, exp_i = self.create_runner_trajs(runner, actions)

        # Batcher
        [env.seed(int(seed)) for env, seed in zip(envs, seeds)]
        runner = PAACRunner(envs)
        batcher = RolloutBatcher(runner, batch_size=self.batch_size)

        action_gen = (a for a in actions)
        action_fn = lambda state, step: next(action_gen)
        for i in range(0, self.num_steps // self.num_envs, horizon):
            batch = batcher.get_batch(select_action_fn=action_fn)

            np.testing.assert_allclose(batch.state_t, exp_s[i : i + horizon])
            np.testing.assert_allclose(batch.reward, exp_r[i : i + horizon])
            np.testing.assert_allclose(batch.done, exp_d[i : i + horizon])
            np.testing.assert_equal(batch.info, exp_i[i : i + horizon])

        runner.close()
        batcher.close()
