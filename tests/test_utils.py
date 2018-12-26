import pytest
import numpy as np
import reward.utils as U


def test_map_range():
    a_map = U.map_range(-1, 1, -1, 2)

    np.testing.assert_allclose(1.25, a_map(.5))
    np.testing.assert_allclose(np.array([1.25, 0.5, -1, 1.625]), a_map(np.array([.5, 0, -1, .75])))
    np.testing.assert_allclose(np.array([[-1, 0.5], [2, 1.25]]), a_map(np.array([[-1, 0], [1, .5]])))