import reward.utils as U
from reward.batcher import ReplayBatcher


class PrReplayBatcher(ReplayBatcher):
    def __init__(self, *args, min_pr=0.01, pr_factor=0.6, is_factor=1., **kwargs):
        self.min_pr = min_pr
        self._pr_factor = pr_factor
        self._is_factor = is_factor
        super().__init__(*args, **kwargs)

    def _create_replay_buffer(self, maxlen):
        return U.buffers.PrReplayBuffer(
            maxlen=maxlen,
            num_envs=self.runner.num_envs,
            min_pr=self.min_pr,
            pr_factor=self._pr_factor,
            is_factor=self._is_factor,
        )

    @property
    def pr_factor(self):
        return self.replay_buffer.get_pr_factor(step=self.num_steps)

    @property
    def is_factor(self):
        return self.replay_buffer.get_is_factor(step=self.num_steps)

    def update_pr(self, idx, pr):
        self.replay_buffer.update_pr(idx=idx, pr=pr, step=self.num_steps)

    def get_is_weight(self, idx):
        return self.replay_buffer.get_is_weight(idx=idx, step=self.num_steps)

    def write_logs(self, logger):
        super().write_logs(logger=logger)
        logger.add_log("ExpReplay/alpha", self.pr_factor)
        logger.add_log("ExpReplay/beta", self.is_factor)
