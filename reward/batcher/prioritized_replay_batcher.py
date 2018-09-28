import reward.utils as U
from reward.batcher import ReplayBatcher


class PrReplayBatcher(ReplayBatcher):
    def __init__(self, *args, min_pr=0.01, pr_factor=0.6, is_factor=1., **kwargs):
        self.min_pr = min_pr
        self.pr_factor = pr_factor
        self.is_factor = is_factor
        super().__init__(*args, **kwargs)

    def _create_replay_buffer(self, maxlen):
        return U.buffers.PrReplayBuffer(
            maxlen=maxlen,
            num_envs=self.runner.num_envs,
            min_pr=self.min_pr,
            pr_factor=self.pr_factor,
            is_factor=self.is_factor,
        )

    def update_pr(self, idx, pr):
        self.replay_buffer.update_pr(idx=idx, pr=pr, step=self.num_steps)

    def get_is_weight(self, idx):
        return self.replay_buffer.get_is_weight(idx=idx, step=self.num_steps)
