import unittest

import numpy as np
from torchrl.utils.memories import RingBuffer
from timer import timeit

MAXLEN = 4


class RingBufferTest(unittest.TestCase):
    @timeit
    def test_rollout(self):
        buffer = RingBuffer(input_shape=(8, 1, 16, 16), maxlen=MAXLEN)

        frame = np.ones((8, 1, 16, 16))
        buffer.append(frame)
        state = buffer.get_data()
        expected = np.zeros((8, 4, 16, 16))
        expected[:, 3, :, :] = np.ones((8, 16, 16))
        self.assertTrue(state.shape[1] == MAXLEN)
        self.assertTrue((state == expected).all())

        frame = 2 * np.ones((8, 1, 16, 16))
        buffer.append(frame)
        state = buffer.get_data()
        expected = np.zeros((8, 4, 16, 16))
        expected[:, 3, :, :] = 2 * np.ones((8, 16, 16))
        expected[:, 2, :, :] = np.ones((8, 16, 16))
        self.assertTrue(state.shape[1] == MAXLEN)
        self.assertTrue((state == expected).all())

        frame = 3 * np.ones((8, 1, 16, 16))
        buffer.append(frame)
        state = buffer.get_data()
        expected = np.zeros((8, 4, 16, 16))
        expected[:, 3, :, :] = 3 * np.ones((8, 16, 16))
        expected[:, 2, :, :] = 2 * np.ones((8, 16, 16))
        expected[:, 1, :, :] = np.ones((8, 16, 16))
        self.assertTrue(state.shape[1] == MAXLEN)
        self.assertTrue((state == expected).all())

        frame = 42 * np.ones((8, 1, 16, 16))
        buffer.append(frame)
        buffer.append(frame)
        buffer.append(frame)
        state = buffer.get_data()
        expected = np.zeros((8, 4, 16, 16))
        expected[:, 3, :, :] = 42 * np.ones((8, 16, 16))
        expected[:, 2, :, :] = 42 * np.ones((8, 16, 16))
        expected[:, 1, :, :] = 42 * np.ones((8, 16, 16))
        expected[:, 0, :, :] = 3 * np.ones((8, 16, 16))
        self.assertTrue(state.shape[1] == MAXLEN)
        self.assertTrue((state == expected).all())
