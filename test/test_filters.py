import pytest
import numpy as np
import torchrl.utils as U
from .timer import timeit


def normalize(x, mean, *, std=None, var=None):
    std = std or np.sqrt(var)
    return (U.to_np(x) - mean) / (std + U.EPSILON)


def scale(x, *, std=None, var=None):
    std = std or np.sqrt(var)
    return U.to_np(x) / (std + U.EPSILON)


def check_meanstd_filter(filter_, data, mean, *, std=None, var=None, rtol=1e-7):
    actual = U.to_np(filter_.normalize(data, add_sample=False))
    expected = normalize(data, mean=mean, std=std, var=var)
    np.testing.assert_allclose(actual, expected)

    actual = U.to_np(filter_.scale(data, add_sample=False))
    expected = scale(data, std=std, var=var)
    np.testing.assert_allclose(actual, expected, rtol=rtol)


@pytest.mark.parametrize("num_features", [1, 5])
def test_meanstd_filter_errors(num_features):
    filter_ = U.filters.MeanStdFilter(num_features=num_features)
    filter_.normalize(np.random.normal(size=(1, num_features)))
    filter_.normalize(np.random.normal(size=(3, num_features)))
    with pytest.raises(ValueError):
        filter_.normalize(np.random.normal(size=(num_features,)))
    with pytest.raises(ValueError):
        filter_.normalize(np.random.normal(size=(4, num_features * 2)))
    with pytest.raises(ValueError):
        filter_.normalize(np.random.normal(size=(1, 1, 1)))

    np.testing.assert_equal(np.array(filter_.xs).shape, (4, num_features))


def test_meanstd_filter_simple():
    filter_ = U.filters.MeanStdFilter(num_features=1, clip_range=np.inf)

    # Test mean and var calculation
    filter_.normalize(np.array([[1]]))
    filter_.normalize(np.array([[8]]))
    filter_.update()
    mean, var = filter_.mean, filter_.var
    expected_mean, expected_var = 4.5, 24.5
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(var, expected_var)

    filter_.normalize(np.array([[9], [5], [3]]))
    filter_.update()
    mean, var = filter_.mean, filter_.var
    expected_mean, expected_var = 5.2, 11.2
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(var, expected_var)

    # Test normalization and scaling
    data = np.array([[7], [42]])
    check_meanstd_filter(
        filter_=filter_, data=data, mean=expected_mean, var=expected_var
    )


@pytest.mark.parametrize("use_latest", [True, False])
def test_meanstd_filter_lazy_transform(use_latest):
    filter_ = U.filters.MeanStdFilter(num_features=1, clip_range=np.inf)

    filter_.normalize(np.array([[1], [12]]))
    filter_.update()

    norm_value = filter_.normalize(
        np.array([[4]]), add_sample=False, use_latest_update=use_latest
    )
    scaled_value = filter_.scale(
        np.array([[4]]), add_sample=False, use_latest_update=use_latest
    )
    expected_norm = normalize(np.array([[4]]), mean=filter_.mean, std=filter_.std)
    expected_scaled = scale(np.array([[4]]), std=filter_.std)
    np.testing.assert_allclose(U.to_np(norm_value), expected_norm)
    np.testing.assert_allclose(U.to_np(scaled_value), expected_scaled)

    filter_.normalize(np.array([[8]]))
    filter_.update()
    if use_latest:
        expected_norm = normalize(np.array([[4]]), mean=filter_.mean, var=filter_.var)
        expected_scaled = scale(np.array([[4]]), std=filter_.std)
    np.testing.assert_allclose(U.to_np(norm_value), expected_norm)
    np.testing.assert_allclose(U.to_np(scaled_value), expected_scaled)


@pytest.mark.parametrize("num_features", [1, 5])
@pytest.mark.parametrize("shift", [0, 10e8])
def test_meanstd_filter_random(num_features, shift):
    update_prob = 0.2

    filter_ = U.filters.MeanStdFilter(num_features=num_features, clip_range=np.inf)

    data = np.random.normal(loc=shift, size=(10000, num_features))
    chunks = 100
    for i, d in enumerate(data.reshape((chunks, -1, num_features))):
        filter_.normalize(d)
        # Random checks
        if np.random.rand() < update_prob and i > 1:
            filter_.update()
            mean, var = filter_.mean, filter_.var
            expected_mean = data[: (i + 1) * chunks].mean(axis=0)
            expected_var = data[: (i + 1) * chunks].var(axis=0, ddof=1)
            np.testing.assert_allclose(mean, expected_mean)
            np.testing.assert_allclose(var, expected_var)

    filter_.update()
    mean, var = filter_.mean, filter_.var
    expected_mean, expected_var = data.mean(axis=0), data.var(axis=0, ddof=1)
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(var, expected_var)

    data = np.random.normal(size=(3, num_features))
    check_meanstd_filter(
        filter_=filter_, data=data, mean=expected_mean, var=expected_var, rtol=1e-7
    )
