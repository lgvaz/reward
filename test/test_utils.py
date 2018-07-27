import pytest
import numpy as np
import torchrl.utils as U


def test_lazy_array_memory_opt():
    arr1 = np.arange(10)
    arr2 = np.arange(10)
    lazy = U.LazyArray([arr1, arr2])
    arr1 += 1

    np.testing.assert_equal(U.to_np(lazy), np.array([arr1, arr2]))


def test_lazy_array_lazy_transform():
    arr1 = np.arange(10)
    arr2 = np.arange(10)
    lazy = U.LazyArray([arr1, arr2], transform=lambda x: x * 0)

    np.testing.assert_equal(U.to_np(lazy), 0)
    # Original data should not be modified
    np.testing.assert_equal(arr1, np.arange(10))
    np.testing.assert_equal(arr2, np.arange(10))
    # LazyArray data should not be modified
    np.testing.assert_equal(lazy.data, [arr1, arr2])
