import pytest
import random
import numpy as np
import reward.utils as U
import reward.batcher.transforms as tfms
from .utils import create_test_array


@pytest.mark.parametrize("num_envs", [1, 4])
def test_state_run_norm(num_envs):
    shape = (num_envs, 4)

    transform = tfms.StateRunNorm()
    filter_ = U.filter.MeanStdFilter(num_features=shape[-1])

    for i in range(10):
        s = np.random.normal(size=shape)
        trans_s = U.to_np(transform.transform_s(s))
        expected = U.to_np(filter_.normalize(s))

        if i == 5:
            transform.transform_batch(batch=None)
            filter_.update()

        np.testing.assert_equal(trans_s, expected)


def test_state_run_scaler_error():
    transform = tfms.StateRunNorm()

    def test_shape_error(shape):
        with pytest.raises(ValueError):
            s = np.random.normal(size=shape)
            transform.transform_s(s)

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
        s = create_test_array(num_envs=num_envs, shape=shape)
        trans_s = U.to_np(transform.transform_s(s))
        buffer.append(s)
        expected = U.to_np(buffer.get_data()).swapaxes(0, 2)[0]

        np.testing.assert_equal(trans_s, expected)
        np.testing.assert_equal(trans_s.shape, expected_shape)


@pytest.mark.parametrize("num_envs", [1, 8])
@pytest.mark.parametrize("shape", [(1, 16, 16), (1, 3, 4), (1, 1, 1)])
def test_frame2float(num_envs, shape):
    transform = tfms.Frame2Float()
    s = np.random.randint(low=0, high=255, size=(num_envs,) + shape, dtype="uint8")

    trans_s = U.to_np(transform.transform_s(s))
    expected = (s / 255.0).astype("float")

    np.testing.assert_equal(trans_s, expected)


def test_reward_const_scaler():
    transform = tfms.RewardConstScaler(factor=0.1)
    r = np.random.normal(size=(32,))
    batch = U.Batch(r=r.copy())

    trans_rew = U.to_np(transform.transform_batch(batch).r)
    expected = r * 0.1

    np.testing.assert_allclose(trans_rew, expected)


@pytest.mark.parametrize("num_envs", [1, 4])
def test_reward_run_scaler(num_envs):
    shape = (5, num_envs)
    transform = tfms.RewardRunScaler()
    filter_ = U.filter.MeanStdFilter(num_features=1)

    for _ in range(10):
        r = np.random.normal(size=shape)
        batch = U.Batch(r=r)
        trans_r = U.to_np(transform.transform_batch(batch).r)
        expected = U.to_np(filter_.scale(r.reshape(-1, 1))).reshape(shape)
        filter_.update()

        np.testing.assert_equal(trans_r, expected)


def test_reward_run_scaler_error():
    transform = tfms.RewardRunScaler()

    def test_shape_error(shape):
        with pytest.raises(ValueError):
            r = np.random.normal(size=shape)
            batch = U.Batch(r=r)
            transform.transform_batch(batch)

    test_shape_error(shape=(1,))
    test_shape_error(shape=(5,))
    test_shape_error(shape=(16, 5, 1))
