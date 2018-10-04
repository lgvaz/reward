import reward.utils as U
from reward.batcher import PrReplayBatcher
from tqdm import tqdm


# TODO: Fix args and kwargs
class DemoReplayBatcher(PrReplayBatcher):
    def __init__(
        self, *args, min_pr=0.01, pr_factor=0.6, is_factor=1., pr_demo=0.3, **kwargs
    ):
        self.pr_demo = pr_demo
        super().__init__(
            *args, min_pr=min_pr, pr_factor=pr_factor, is_factor=is_factor, **kwargs
        )

    def _create_replay_buffer(self, maxlen):
        return U.buffers.DemoReplayBuffer(
            maxlen=maxlen, num_envs=self.runner.num_envs, pr_demo=self.pr_demo
        )

    def populate_expert(self, n=None, pct=None, act_fn=None):
        assert (n and not pct) or (pct and not n)
        num_replays = int(n or pct * self.replay_buffer.maxlen)

        state_t = self.runner.reset()
        state_t = self.transform_state(state_t)

        tqdm.write("Populating Replay Buffer...")
        for _ in tqdm(range(num_replays)):
            if act_fn is not None:
                action = act_fn(state=U.to_tensor(state_t), step=0)
            else:
                action = self.runner.sample_random_action()
            state_tp1, reward, done, info = self.runner.act(action)
            state_tp1 = self.transform_state(state_tp1)

            self.replay_buffer.add_sample_demo(
                state=state_t,
                action=action,
                reward=reward,
                done=done,
                # info=info,
            )

            state_t = state_tp1
