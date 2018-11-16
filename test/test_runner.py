import pytest
import numpy as np

from reward.env import GymEnv, AtariEnv
from reward.env.wrappers import RGB2GRAY, Rescale
from reward.runner import SingleRunner, PAACRunner

NUM_STEPS = 1000
NUM_ENVS = 4
SEED = np.random.choice(4200)


def create_env(num_envs):
    env = []
    env.append([GymEnv("CartPole-v1") for _ in range(num_envs)])
    env.append([GymEnv("Pendulum-v0") for _ in range(num_envs)])
    env.append(
        [
            Rescale(RGB2GRAY(AtariEnv("PongNoFrameskip-v4")), shape=(84, 84))
            for _ in range(num_envs)
        ]
    )

    return env


@pytest.mark.parametrize("env", create_env(num_envs=1))
def test_single_runner(env):
    env = env[0]
    actions = [env.sample_random_action() for _ in range(NUM_STEPS)]
    runner = SingleRunner(env)

    env.seed(SEED)
    exp_s, exp_r, exp_d, exp_i = create_expected_trajs(env, actions)
    env.seed(SEED)
    states, rs, dones, infos = create_runner_trajs(runner, actions)

    np.testing.assert_allclose(states, exp_s[:, None])
    np.testing.assert_allclose(rs, exp_r[:, None])
    np.testing.assert_allclose(dones, exp_d[:, None])
    np.testing.assert_equal(infos, exp_i)

    runner.close()


@pytest.mark.parametrize("env", create_env(num_envs=NUM_ENVS))
def test_paac_runner(env):
    seeds = np.random.choice(4200, NUM_ENVS)
    actions = np.array(
        [
            [env[0].sample_random_action() for _ in range(NUM_STEPS)]
            for _ in range(NUM_ENVS)
        ]
    )

    # Expected
    [env.seed(int(seed)) for env, seed in zip(env, seeds)]
    exp_s, exp_r, exp_d, exp_i = [
        np.array(a).swapaxes(0, 1)
        for a in zip(*[create_expected_trajs(env, a) for env, a in zip(env, actions)])
    ]

    # Runner
    [env.seed(int(seed)) for env, seed in zip(env, seeds)]
    runner = PAACRunner(env)
    states, rs, dones, infos = create_runner_trajs(runner, actions.swapaxes(0, 1))

    np.testing.assert_allclose(states, exp_s)
    np.testing.assert_allclose(rs, exp_r)
    np.testing.assert_allclose(dones, exp_d)
    np.testing.assert_equal(infos, exp_i)

    runner.close()


def create_expected_trajs(env, actions):
    states, rs, dones, infos = [], [], [], []

    state = env.reset()
    for a in actions:
        next_state, reward, done, info = env.step(a)

        states.append(state)
        rs.append(reward)
        dones.append(done)
        infos.append(info)

        if done:
            next_state = env.reset()

        state = next_state

    return list(map(np.array, [states, rs, dones, infos]))


def create_runner_trajs(runner, actions):
    states, rs, dones, infos = [], [], [], []

    state = runner.reset()
    for a in actions:
        next_state, reward, done, info = runner.act(a)

        states.append(state)
        rs.append(reward)
        dones.append(done)
        infos.append(info)

        state = next_state

    return list(map(np.array, [states, rs, dones, infos]))
