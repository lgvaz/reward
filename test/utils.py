import numpy as np
from functools import wraps


def before_after_equal(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        v = func(*args, **kwargs)
        v_copy = v.copy()

        yield v

        np.testing.assert_equal(v, v_copy, err_msg='array should not be modified')

    return wrapper


def create_test_array(num_envs, shape=()):
    return np.array([np.random.uniform(size=shape) for i in range(num_envs)])
