import pytest
import random
import numpy as np
import torchrl.utils as U
import torchrl.batchers.transforms as tfms
from .utils import create_test_array


@pytest.mark.parametrize("num_envs", [1, 4])
def test_state_run_norm(num_envs):
    shape = (num_envs, 4)

    transform = tfms.StateRunNorm()
    filter_ = U.filters.MeanStdFilter(num_features=shape[-1])

    for i in range(10):
        state = np.random.normal(size=shape)
        trans_state = U.to_np(transform.transform_state(state))
        expected = U.to_np(filter_.normalize(state))

        if i == 5:
            transform.transform_batch(batch=None)
            filter_.update()

        np.testing.assert_equal(trans_state, expected)


def test_state_run_scaler_error():
    transform = tfms.StateRunNorm()

    def test_shape_error(shape):
        with pytest.raises(ValueError):
            state = np.random.normal(size=shape)
            transform.transform_state(state)

    test_shape_error(shape=(1,))
    test_shape_error(shape=(5,))
    test_shape_error(shape=(16, 5, 1))


@pytest.mark.parametrize("num_envs", [1, 8])
@pytest.mark.parametrize("shape", [(1, 16, 16), (1, 3, 4), (1, 1, 1)])
def test_stack_states(num_envs, shape, maxlen=3):
    transform = tfms.StackStates(n=maxlen)
    buffer = U.buffers.RingBuffer(input_shape=(num_envs,) + shape, maxlen=maxlen)
    expected_shape = (num_envs, maxlen) + shape[1:]

    for _ in range(100):
        state = create_test_array(num_envs=num_envs, shape=shape)
        trans_state = U.to_np(transform.transform_state(state))
        buffer.append(state)
        expected = U.to_np(buffer.get_data()).swapaxes(0, 2)[0]

        np.testing.assert_equal(trans_state, expected)
        np.testing.assert_equal(trans_state.shape, expected_shape)


@pytest.mark.parametrize("num_envs", [1, 8])
@pytest.mark.parametrize("shape", [(1, 16, 16), (1, 3, 4), (1, 1, 1)])
def test_frame2float(num_envs, shape):
    transform = tfms.Frame2Float()
    state = np.random.randint(low=0, high=255, size=(num_envs,) + shape, dtype="uint8")

    trans_state = U.to_np(transform.transform_state(state))
    expected = (state / 255.).astype("float")

    np.testing.assert_equal(trans_state, expected)


def test_reward_const_scaler():
    transform = tfms.RewardConstScaler(factor=0.1)
    reward = np.random.normal(size=(32,))
    batch = U.Batch(reward=reward.copy())

    trans_rew = U.to_np(transform.transform_batch(batch).reward)
    expected = reward * 0.1

    np.testing.assert_allclose(trans_rew, expected)


@pytest.mark.parametrize("num_envs", [1, 4])
def test_reward_run_scaler(num_envs):
    shape = (5, num_envs)
    transform = tfms.RewardRunScaler()
    filter_ = U.filters.MeanStdFilter(num_features=1)

    for _ in range(10):
        reward = np.random.normal(size=shape)
        batch = U.Batch(reward=reward)
        trans_reward = U.to_np(transform.transform_batch(batch).reward)
        expected = U.to_np(filter_.scale(reward.reshape(-1, 1))).reshape(shape)
        filter_.update()

        np.testing.assert_equal(trans_reward, expected)


def test_reward_run_scaler_error():
    transform = tfms.RewardRunScaler()

    def test_shape_error(shape):
        with pytest.raises(ValueError):
            reward = np.random.normal(size=shape)
            batch = U.Batch(reward=reward)
            transform.transform_batch(batch)

    test_shape_error(shape=(1,))
    test_shape_error(shape=(5,))
    test_shape_error(shape=(16, 5, 1))
