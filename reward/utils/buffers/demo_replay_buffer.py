import numpy as np
from reward.utils.buffers import PrReplayBuffer


class DemoReplayBuffer(PrReplayBuffer):
    def __init__(
        self, maxlen, num_envs, min_pr=0.01, pr_factor=0.6, is_factor=1., pr_demo=0.3
    ):
        super().__init__(maxlen, num_envs, min_pr=0.01, pr_factor=0.6, is_factor=1.)
        self.pr_demo = pr_demo
        self.start_idx = 0
        self.pr_demos = np.zeros(maxlen)

    def add_sample_demo(self, **kwargs):
        super().add_sample(**kwargs)
        self.pr_demos[self.current_idx] = self.pr_demo
        self.start_idx += 1

    def add_sample(self, **kwargs):
        if self.current_idx == self.maxlen - 1:
            self.current_idx = self.start_idx
        super().add_sample(**kwargs)

    def update_pr(self, idx, pr, step):
        pr = pr.squeeze()
        pr = pr + self.get_min_pr(step) + self.pr_demos[idx]
        self.probs[idx] = pr ** self.get_pr_factor(step)
