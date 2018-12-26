import pytest
import numpy as np
import reward.utils as U


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


@pytest.mark.parametrize("n_features", [1, 5])
def test_meanstd_filter_errors(n_features):
    filter_ = U.filter.MeanStdFilter(n_features=n_features)
    filter_.normalize(np.random.normal(size=(1, n_features)))
    filter_.normalize(np.random.normal(size=(3, n_features)))
    with pytest.raises(ValueError): filter_.normalize(np.random.normal(size=(n_features,)))
    with pytest.raises(ValueError): filter_.normalize(np.random.normal(size=(4, n_features * 2)))
    with pytest.raises(ValueError): filter_.normalize(np.random.normal(size=(1, 1, 1)))
    np.testing.assert_equal(np.array(filter_.xs).shape, (4, n_features))

def test_meanstd_filter_simple():
    filter_ = U.filter.MeanStdFilter(n_features=1, clip_range=np.inf)
    # Test mean and var calculation
    filter_.normalize(np.array([[1]]))
    filter_.normalize(np.array([[8]]))
    filter_.update()
    expected_mean, expected_var = 4.5, 24.5
    np.testing.assert_allclose(filter_.mean, expected_mean)
    np.testing.assert_allclose(filter_.var, expected_var)

    filter_.normalize(np.array([[9], [5], [3]]))
    filter_.update()
    expected_mean, expected_var = 5.2, 11.2
    np.testing.assert_allclose(filter_.mean, expected_mean)
    np.testing.assert_allclose(filter_.var, expected_var)
    # Test normalization and scaling
    data = np.array([[7], [42]])
    check_meanstd_filter(filter_=filter_, data=data, mean=expected_mean, var=expected_var)

@pytest.mark.parametrize("n_features", [1, 5])
@pytest.mark.parametrize("shift", [0, 10e8])
def test_meanstd_filter_random(n_features, shift):
    update_prob = 0.2
    filter_ = U.filter.MeanStdFilter(n_features=n_features, clip_range=np.inf)
    data = np.random.normal(loc=shift, size=(10000, n_features))
    chunks = 100
    for i, d in enumerate(data.reshape((chunks, -1, n_features))):
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
    data = np.random.normal(size=(3, n_features))
    check_meanstd_filter(filter_=filter_, data=data, mean=expected_mean, var=expected_var, rtol=1e-6)
