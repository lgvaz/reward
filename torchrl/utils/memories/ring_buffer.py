import numpy as np


class RingBuffer:
    '''
    Ring buffer used for stacking game frames, original implementation of this particular
    buffer can be found here `<here https://github.com/Alfredvc/paac/blob/HEAD/environment.py#L58-L58>`_.
    '''

    def __init__(self, *, input_shape, maxlen):
        assert len(input_shape) == 4, 'data have {} dims and must have 4 (NCHW)'.format(
            len(input_shape))
        assert input_shape[1] == 1, 'data number of channels is {} and must be 1'.format(
            len(input_shape))

        self.maxlen = maxlen
        self.input_shape = list(input_shape)
        self.input_shape[1] = self.maxlen

        self.circular_idx = self._create_circular_indices()
        self.current_idx = 0

        self.reset()

    def _create_circular_indices(self):
        seq = list(range(self.maxlen))
        idxs = []

        for i in range(self.maxlen):
            n = i % len(seq)
            idxs.append(seq[n:] + seq[:n])

        return idxs

    def reset(self):
        self.data = np.zeros(self.input_shape)

    def append(self, data):
        assert data.ndim == 4, 'data have {} dims and must have 4 (NCHW)'.format(
            data.ndim)
        assert data.shape[1] == 1, 'data number of channels is {} and must be 1'.format(
            data.shape[1])
        self.data[:, self.current_idx, :, :] = data[:, 0, :, :]
        self.current_idx = (self.current_idx + 1) % self.maxlen

    def get_data(self):
        #TODO: Need copy?
        return self.data[:, self.circular_idx[self.current_idx], :, :]
