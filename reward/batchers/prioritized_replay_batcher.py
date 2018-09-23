import reward.utils as U
from reward.batchers import ReplayBatcher


class PrReplayBatcher(ReplayBatcher):
    def __init__(self, *args, min_pr=0.01, pr_weight=0.6, **kwargs):
        self.min_pr = min_pr
        self.pr_weight = pr_weight
        super().__init__(*args, **kwargs)

    def _create_replay_buffer(self, maxlen):
        return U.buffers.PrReplayBuffer(
            maxlen=maxlen,
            num_envs=self.runner.num_envs,
            min_pr=self.min_pr,
            pr_weight=self.pr_weight,
        )

    def update_pr(self, idx, pr):
        self.replay_buffer.update_pr(idx=idx, pr=pr, step=self.num_steps)
