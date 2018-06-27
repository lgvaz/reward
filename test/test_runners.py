import unittest
import numpy as np

from torchrl.envs import GymEnv, AtariEnv
from torchrl.envs.wrappers import RGB2GRAY, Rescale
from torchrl.runners import SingleRunner, PAACRunner
from .timer import timeit


class TestRunners(unittest.TestCase):
    def setUp(self):
        self.num_steps = 1000
        self.seed = np.random.choice(4200)

    @timeit
    def test_single_runner(self):
        self._test_single_runner(GymEnv('CartPole-v1'))
        self._test_single_runner(GymEnv('Pendulum-v0'))
        self._test_single_runner(
            Rescale(RGB2GRAY(AtariEnv('PongNoFrameskip-v4')), shape=(84, 84)))

    @timeit
    def test_paac_runner(self):
        self.num_envs = 16

        self._test_paac_runner([GymEnv('CartPole-v1') for _ in range(self.num_envs)])
        self._test_paac_runner([GymEnv('Pendulum-v0') for _ in range(self.num_envs)])
        self._test_paac_runner([
            Rescale(RGB2GRAY(AtariEnv('PongNoFrameskip-v4')), shape=(84, 84))
            for _ in range(self.num_envs)
        ])

    def _test_single_runner(self, env):
        actions = [env.sample_random_action() for _ in range(self.num_steps)]
        runner = SingleRunner(env)

        env.seed(self.seed)
        exp_s, exp_r, exp_d, exp_i = self.create_expected_trajs(env, actions)
        env.seed(self.seed)
        states, rewards, dones, infos = self.create_runner_trajs(runner, actions)

        np.testing.assert_allclose(states, exp_s[:, None])
        np.testing.assert_allclose(rewards, exp_r[:, None])
        np.testing.assert_allclose(dones, exp_d[:, None])
        np.testing.assert_equal(infos, exp_i)

        runner.close()

    def _test_paac_runner(self, envs):
        seeds = np.random.choice(4200, self.num_envs)
        actions = np.array(
            [[envs[0].sample_random_action() for _ in range(self.num_steps)]
             for _ in range(self.num_envs)])

        # Expected
        [env.seed(int(seed)) for env, seed in zip(envs, seeds)]
        exp_s, exp_r, exp_d, exp_i = [
            np.array(a).swapaxes(0, 1)
            for a in zip(
                *[self.create_expected_trajs(env, a) for env, a in zip(envs, actions)])
        ]

        # Runner
        [env.seed(int(seed)) for env, seed in zip(envs, seeds)]
        runner = PAACRunner(envs)
        states, rewards, dones, infos = self.create_runner_trajs(
            runner, actions.swapaxes(0, 1))

        np.testing.assert_allclose(states, exp_s)
        np.testing.assert_allclose(rewards, exp_r)
        np.testing.assert_allclose(dones, exp_d)
        np.testing.assert_equal(infos, exp_i)

        runner.close()

    def create_expected_trajs(self, env, actions):
        states, rewards, dones, infos = [], [], [], []

        state = env.reset()
        for a in actions:
            next_state, reward, done, info = env.step(a)

            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

            if done:
                next_state = env.reset()

            state = next_state

        return list(map(np.array, [states, rewards, dones, infos]))

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
