import pdb
import numpy as np
import reward
from reward.utils.buffers import ReplayBuffer
from reward.utils import make_callable


# TODO: Why paper suggest using SumTree instead of this approach I did?
class PrReplayBuffer(ReplayBuffer):
    def __init__(self, maxlen, num_envs, min_pr=0.01, pr_factor=0.6, is_factor=1.):
        """
        Parameters
        ----------
        min_pr: float or schedule
            Minimum priority possible (epsilon in the paper).
        pr_factor: float or schedule
            Determines how much prioritization is used (alpha in the paper).
        is_factor: float or schedule
            Importance sampling weight for correcting the bias (beta in the paper).
        """
        super().__init__(maxlen=maxlen, num_envs=num_envs)
        self._min_pr = make_callable(min_pr)
        self._pr_factor = make_callable(pr_factor)
        self._is_factor = make_callable(is_factor)

        self._probs = np.ones(self.maxlen)

    @property
    def probs(self):
        return self._probs[: len(self)]

    def get_min_pr(self, step):
        return self._min_pr(step)

    def get_pr_factor(self, step):
        return self._pr_factor(step)

    def get_is_factor(self, step):
        return self._is_factor(step)

    def add_sample(self, **kwargs):
        super().add_sample(**kwargs)
        # New transition start with max probability
        prob = np.max(self.probs)
        self.probs[self.current_idx] = prob

    def sample(self, batch_size):
        # The sum of all probs should be 1
        probs = self.probs[: self.available_idxs]
        probs = probs / np.sum(probs)
        idxs = np.random.choice(self.available_idxs, batch_size, replace=False, p=probs)

        return self._get_batch(idxs=idxs)

    def get_is_weight(self, idx, step):
        probs = self.probs[idx]
        probs = probs / np.sum(probs)
        is_weights = (len(self) * probs) ** -self.get_is_factor(step)

        return is_weights[:, None]

    def update_pr(self, idx, pr, step):
        pr = pr.squeeze()
        self.probs[idx] = (pr + self.get_min_pr(step)) ** self.get_pr_factor(step)
