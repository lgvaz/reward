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
    acs = [env.sample_random_ac() for _ in range(NUM_STEPS)]
    runner = SingleRunner(env)

    env.seed(SEED)
    exp_s, exp_r, exp_d, exp_i = create_expected_trajs(env, acs)
    env.seed(SEED)
    ss, rs, ds, infos = create_runner_trajs(runner, acs)

    np.testing.assert_allclose(ss, exp_s[:, None])
    np.testing.assert_allclose(rs, exp_r[:, None])
    np.testing.assert_allclose(ds, exp_d[:, None])
    np.testing.assert_equal(infos, exp_i)

    runner.close()


@pytest.mark.parametrize("env", create_env(num_envs=NUM_ENVS))
def test_paac_runner(env):
    seeds = np.random.choice(4200, NUM_ENVS)
    acs = np.array(
        [[env[0].sample_random_ac() for _ in range(NUM_STEPS)] for _ in range(NUM_ENVS)]
    )

    # Expected
    [env.seed(int(seed)) for env, seed in zip(env, seeds)]
    exp_s, exp_r, exp_d, exp_i = [
        np.array(a).swapaxes(0, 1)
        for a in zip(*[create_expected_trajs(env, a) for env, a in zip(env, acs)])
    ]

    # Runner
    [env.seed(int(seed)) for env, seed in zip(env, seeds)]
    runner = PAACRunner(env)
    ss, rs, ds, infos = create_runner_trajs(runner, acs.swapaxes(0, 1))

    np.testing.assert_allclose(ss, exp_s)
    np.testing.assert_allclose(rs, exp_r)
    np.testing.assert_allclose(ds, exp_d)
    np.testing.assert_equal(infos, exp_i)

    runner.close()


def create_expected_trajs(env, acs):
    ss, rs, ds, infos = [], [], [], []

    state = env.reset()
    for a in acs:
        sn, r, d, info = env.step(a)

        ss.append(state)
        rs.append(r)
        ds.append(d)
        infos.append(info)

        if d:
            sn = env.reset()

        state = sn

    return list(map(np.array, [ss, rs, ds, infos]))


def create_runner_trajs(runner, acs):
    ss, rs, ds, infos = [], [], [], []

    state = runner.reset()
    for a in acs:
        sn, r, d, info = runner.act(a)

        ss.append(state)
        rs.append(r)
        ds.append(d)
        infos.append(info)

        state = sn

    return list(map(np.array, [ss, rs, ds, infos]))
