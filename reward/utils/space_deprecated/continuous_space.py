import numpy as np
from reward.utils.space import BaseSpace


class Continuous(BaseSpace):
    def __init__(self, low=None, high=None, shape=None):
        low, high = np.array(low), np.array(high)
        assert low.shape == high.shape if shape is None else True
        shape = shape or low.shape
        super().__init__(shape=shape, dtype=np.float32)

        self.low = low + np.zeros(self.shape, dtype=self.dtype)
        self.high = high + np.zeros(self.shape, dtype=self.dtype)

    def __repr__(self):
        info = "shape={},low={},high={}".format(self.shape, self.low, self.high)
        return "Continuous({})".format(info)

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high, size=self.shape)
