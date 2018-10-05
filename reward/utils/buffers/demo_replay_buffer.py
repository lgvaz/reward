import numpy as np
from reward.utils.buffers import PrReplayBuffer
from reward.utils import make_callable


class DemoReplayBuffer(PrReplayBuffer):
    def __init__(
        self, maxlen, num_envs, *, pr_factor, is_factor, min_pr=0.01, pr_demo=1.
    ):
        super().__init__(
            maxlen, num_envs, pr_factor=pr_factor, is_factor=is_factor, min_pr=min_pr
        )
        self._pr_demo = make_callable(pr_demo)
        self.start_idx = 0
        self.pr_demos = np.zeros(maxlen)

    def get_pr_demo(self, step):
        return self._pr_demo(step)

    def add_sample_demo(self, **kwargs):
        super().add_sample(**kwargs)
        self.pr_demos[self.current_idx] = self.get_pr_demo(step=0)
        self.start_idx += 1

    def add_sample(self, **kwargs):
        if self.current_idx == self.maxlen - 1:
            self.current_idx = self.start_idx
        super().add_sample(**kwargs)

    def update_pr(self, idx, pr, step):
        pr = pr.squeeze()
        # self.pr_demos[: self.start_idx] = self.get_pr_demo(step)
        self.pr_demos[: self.start_idx] = np.max(self.probs) * self.get_pr_demo(step)
        pr = pr + self.get_min_pr(step) + self.pr_demos[idx]
        self.probs[idx] = pr ** self.get_pr_factor(step)
