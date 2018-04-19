import numpy as np
import pickle


#TODO: Change docstring
class Normalizer(object):
    """
    From: `pat-coady <https://github.com/pat-coady/trpo/blob/master/src/utils.py#L13>`_.
    Generate scale and offset based on running mean and stddev along axis=0.

    offset = running mean

    scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)

    Parameters
    ----------
    shape: tuple
        Shape of the variable that will be normalized.
    """

    def __init__(self, shape, path=None):
        self.path = path
        self.vars = np.zeros(shape)
        self.means = np.zeros(shape)
        self.xs = []
        self.m = 0
        self.n = 0
        self.first_pass = True

    def update(self):
        '''
        Update running mean and variance (this is an exact method).
        '''
        xs = np.array(self.xs)
        self.xs = []
        if self.first_pass:
            self.means = np.mean(xs, axis=0)
            self.vars = np.var(xs, axis=0)
            self.m = xs.shape[0]
            self.first_pass = False
        else:
            n = xs.shape[0]
            new_data_var = np.var(xs, axis=0)
            new_data_mean = np.mean(xs, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) /
                         (self.m + n) - np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n

    def get(self):
        '''
        Returns the current scaling factor and offset.

        Returns
        -------
        scale: numpy.ndarray
            The current scaling factor.
        offset: numpy.ndarray
            The current offset.
        '''
        return 1 / (np.sqrt(self.vars) + 0.1) / 3, self.means

    def normalize(self, x):
        '''
        Normalizes the input using the running mean and stddev.

        Returns
        -------
        numpy.ndarray
            The normalized input.
        '''
        self.xs.append(x)
        scale, offset = self.get()
        return (x - offset) * scale

    def scale(self, x):
        self.xs.append(x)
        scale, _ = self.get()
        return x * scale

    # TODO: Re-work this
    def save(self):
        assert self.path is not None, 'You must define a path when creating this object'
        with open(self.path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    # TODO: Re-work this
    @classmethod
    def initialize_or_load(cls, obs_dim, path=None):
        # TODO: Substitute for JSON loader
        scaler = cls(obs_dim, path)
        if path is not None and os.path.exists(path):
            with open(path, 'rb') as f:
                attributes = pickle.load(f)
                scaler.__dict__.update(attributes)

        return scaler
