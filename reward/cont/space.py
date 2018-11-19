import numpy as np


class Space():
    def __init__(self, low=None, high=None, shape=None):
        low, high = np.array(low), np.array(high)
        assert low.shape == high.shape if shape is None else True
        shape = shape or low.shape
        self.shape, self.dtype = shape, np.float32
        self.low = low + np.zeros(self.shape, dtype=self.dtype)
        self.high = high + np.zeros(self.shape, dtype=self.dtype)

    def __repr__(self):
        return f'Continuous(shape={self.shape},low={self.low},high={self.high})'

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high, size=self.shape)
