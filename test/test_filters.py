import numpy as np
import torchrl.utils as U
from .timer import timeit


def test_meanstd_filter_simple():
    filter_ = U.filters.MeanStdFilter(shape=1)

    filter_.normalize([1])
    filter_.normalize([8])
    filter_.update()
    mean, var = filter_.mean, filter_.var
    expected_mean, expected_var = 4.5, 24.5
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(var, expected_var)

    filter_.normalize([9])
    filter_.normalize([5])
    filter_.normalize([3])
    filter_.update()
    mean, var = filter_.mean, filter_.var
    expected_mean, expected_var = 5.2, 11.2
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(var, expected_var)


def test_filter_random():
    shape = (3, 5)
    update_prob = 0.1

    filter_ = U.filters.MeanStdFilter(shape=shape)

    data = np.random.normal(size=(10000, ) + shape)
    for i, d in enumerate(data):
        filter_.normalize(d[None])
        # Random checks
        if np.random.rand() < update_prob and i > 1:
            filter_.update()
            mean, var = filter_.mean, filter_.var
            expected_mean = data[:i + 1].mean(axis=0)
            expected_var = data[:i + 1].var(axis=0, ddof=1)
            np.testing.assert_allclose(mean, expected_mean)
            np.testing.assert_allclose(var, expected_var)

    filter_.update()
    mean, var = filter_.mean, filter_.var
    expected_mean, expected_var = data.mean(axis=0), data.var(axis=0, ddof=1)
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(var, expected_var)


def test_filter_huge():
    shape = (3, 5)
    update_prob = 0.1

    filter_ = U.filters.MeanStdFilter(shape=shape)

    data = 10e7 + np.random.normal(size=(10000, ) + shape)
    for i, d in enumerate(data):
        filter_.normalize(d[None])
        # Random checks
        if np.random.rand() < update_prob and i > 1:
            filter_.update()
            mean, var = filter_.mean, filter_.var
            expected_mean = data[:i + 1].mean(axis=0)
            expected_var = data[:i + 1].var(axis=0, ddof=1)
            np.testing.assert_allclose(mean, expected_mean)
            np.testing.assert_allclose(var, expected_var)

    filter_.update()
    mean, var = filter_.mean, filter_.var
    expected_mean, expected_var = data.mean(axis=0), data.var(axis=0, ddof=1)
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(var, expected_var)
