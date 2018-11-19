import numpy as np
from collections import deque


class RingBuffer:
    def __init__(self, in_sz, maxlen):
        if not maxlen > 1:
            raise ValueError("maxlen is {} and must be greater than 1".format(maxlen))
        self.in_sz, self.maxlen = in_sz, maxlen
        self.buffer = deque(maxlen=maxlen)
        self.reset()

    def reset(self):
        data = np.zeros(self.in_sz)
        for _ in range(self.maxlen): self.append(data)

    def append(self, data): self.buffer.append(data)

    def get(self): return self.buffer
