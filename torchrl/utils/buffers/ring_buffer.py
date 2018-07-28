import numpy as np
from collections import deque
from torchrl.utils import LazyArray


class RingBuffer:
    def __init__(self, *, input_shape, maxlen):
        if not maxlen > 1:
            raise ValueError("maxlen is {} and must be greater than 1".format(maxlen))
        self.input_shape = input_shape
        self.maxlen = maxlen

        self.buffer = deque(maxlen=maxlen)
        self.reset()

    def reset(self):
        data = np.zeros(self.input_shape)
        for _ in range(self.maxlen):
            self.append(data)

    def append(self, data):
        self.buffer.append(data)

    def get_data(self):
        return LazyArray(list(self.buffer))
