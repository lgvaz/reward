import numpy as np
from reward.utils import EPSILON


class MeanStdFilter:
    "Calculates the exact mean and std deviation, originally by `ray_rllib <https://goo.gl/fMv49b>`_."
    def __init__(self, n_features, clip_range=np.inf):
        if not isinstance(n_features, int): raise ValueError('n_features should be an int, got {}'.format(n_features))
        self.n_features, self.clip_range, self.n, self.xs = n_features, clip_range, 0, []
        self.M, self.S = np.zeros(n_features), np.zeros(n_features)

    def normalize(self, x, add_sample=True):
        "Normalizes x by subtracting the mean and dividing by the standard deviation."
        self._check_shape(x)
        if add_sample: self.xs.extend(x)
        return ((x - self.mean) / (self.std + EPSILON)).clip(-self.clip_range, self.clip_range)

    def scale(self, x, add_sample=True):
        "Scales x by dividing by the standard deviation."
        self._check_shape(x)
        if add_sample: self.xs.extend(x)
        return (x / (self.std + EPSILON)).clip(-self.clip_range, self.clip_range)

    def update(self):
        n_old, n_new = self.n, len(self.xs)
        if n_new == 0: return
        x, self.xs = np.array(self.xs), []
        self.n += n_new
        x_mean = x.mean(axis=0)
        x_std  = ((x - x_mean) ** 2).sum(axis=0)
        # First update
        if self.n == n_new:
            self.S[:] = x_std
            self.M[:] = x_mean
        else:
            new_mean = (n_old * self.M + n_new * x_mean) / self.n
            self.S[:] = self.S + x_std + (self.M - x_mean) ** 2 * n_old * n_new / self.n
            self.M[:] = new_mean

    @property
    def mean(self): return self.M
    @property
    def std(self): return np.sqrt(self.var)
    @property
    def var(self):
        if self.n == 0 or self.n == 1: return np.ones(self.S.shape)
        else:                          return self.S / (self.n - 1)

    def _check_shape(self, x):
        if not (self.n_features,) == x.shape[1:]:
            raise ValueError(f'Data shape must be (num_samples, {self.n_features}) but is {x.shape}')
