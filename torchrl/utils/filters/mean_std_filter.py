import warnings
import numpy as np
from torchrl.utils import EPSILON, to_np, LazyArray


class MeanStdFilter:
    """
    Calculates the exact mean and std deviation, originally by `ray_rllib
    <https://goo.gl/fMv49b>`_.

    Returning a LazyArray means that the normalization/scaling is always done using
    the latest values for the mean and std_dev.

    Parameters
    ----------
    shape: tuple
        The shape of the inputs to :meth:`self.normalize` and :meth:`self.scale`
    clip_range: float
        The output of :meth:`self.normalize` and :meth:`self.scale` will be clipped
        to this range, use np.inf for no clipping. (Default is 5)
    """

    def __init__(self, num_features, clip_range=5.):
        # if len(shape) > 2:
        #     warnings.warn(
        #         UserWarning('Input shape should be in the form (num_envs, num_features), '
        #                     'might not work as expected with different shapes.'))
        if not isinstance(num_features, int):
            raise ValueError(
                "num_features should be an int, got {}".format(num_features)
            )

        self.num_features = num_features
        self.clip_range = clip_range
        self.n = 0
        self.xs = []

        self.M = np.zeros(num_features)
        self.S = np.zeros(num_features)

    def _norm(self, use_latest):
        mean_copy = None if use_latest else self.mean.copy()
        std_copy = None if use_latest else self.std.copy()

        def apply(x):
            mean = self.mean if use_latest else mean_copy
            std = self.std if use_latest else std_copy
            return ((x - mean) / (std + EPSILON)).clip(
                min=-self.clip_range, max=self.clip_range
            )

        return apply

    def _scale(self, use_latest):
        std_copy = None if use_latest else self.std.copy()

        def apply(x):
            std = self.std if use_latest else std_copy
            return (x / (std + EPSILON)).clip(min=-self.clip_range, max=self.clip_range)

        return apply

    def _check_shape(self, x):
        if not (self.num_features,) == x.shape[1:]:
            raise ValueError(
                "Data shape must be (num_samples, {}) but is {}".format(
                    self.num_features, x.shape
                )
            )

    @property
    def mean(self):
        return self.M

    @property
    def var(self):
        if self.n == 0 or self.n == 1:
            return np.ones(self.S.shape)
        else:
            return self.S / (self.n - 1)

    @property
    def std(self):
        return np.sqrt(self.var)

    def update(self):
        n_old = self.n
        n_new = len(self.xs)
        if n_new == 0:
            return

        x = to_np(self.xs)
        self.n += n_new
        self.xs = []

        x_mean = x.mean(axis=0)
        x_std = ((x - x_mean) ** 2).sum(axis=0)
        # First update
        if self.n == n_new:
            self.M[:] = x_mean
            self.S[:] = x_std
        else:
            new_mean = (n_old * self.M + n_new * x_mean) / self.n
            self.S[:] = self.S + x_std + (self.M - x_mean) ** 2 * n_old * n_new / self.n
            self.M[:] = new_mean

    def normalize(self, x, add_sample=True, use_latest_update=False):
        """
        Normalizes x by subtracting the mean and dividing by the standard deviation.

        Parameters
        ----------
        add_sample: bool
            If True x will be added as a new sample and will be considered when
            the filter is updated via :meth:`self.update`. (Default is True)
        use_latest_update: bool
            If False the current value of the mean and std is used for normalization,
            if True the values used will be the ones available when calling np.array
            on the returned LazyArray (which will be the most updated values for
            the filter). (Default is False)

        Returns
        -------
        LazyArray
        """
        self._check_shape(x)
        if add_sample:
            self.xs.extend(x)
        return LazyArray(x, transform=self._norm(use_latest=use_latest_update))

    def scale(self, x, add_sample=True, use_latest_update=False):
        """
        Scales x by dividing by the standard deviation.

        Parameters
        ----------
        add_sample: bool
            If True x will be added as a new sample and will be considered when
            the filter is updated via :meth:`self.update`. (Default is True)
        use_latest_update: bool
            If False the current value of std is used for scaling, if True the
            values used will be the ones available when calling np.array on the
            returned LazyArray (which will be the most updated values for the
            filter). (Default is False)

        Returns
        -------
        LazyArray
        """
        self._check_shape(x)
        if add_sample:
            self.xs.extend(x)
        return LazyArray(x, transform=self._scale(use_latest=use_latest_update))
