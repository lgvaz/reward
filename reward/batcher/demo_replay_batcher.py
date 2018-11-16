# import reward.utils as U
# from tqdm import tqdm
# from reward.batcher import PrReplayBatcher
# from reward.utils.buffers import DemoReplayBuffer


# # TODO: Fix args and kwargs
# class DemoReplayBatcher(PrReplayBatcher):
#     def __init__(
#         self,
#         *args,
#         min_pr=0.01,
#         pr_factor=0.6,
#         is_factor=1.,
#         pr_demo=0.3,
#         replay_buffer_fn=DemoReplayBuffer,
#         **kwargs
#     ):
#         self.pr_demo = pr_demo
#         super().__init__(
#             *args,
#             min_pr=min_pr,
#             pr_factor=pr_factor,
#             is_factor=is_factor,
#             replay_buffer_fn=replay_buffer_fn,
#             **kwargs
#         )

#     def _create_replay_buffer(self, replay_buffer_fn):
#         return replay_buffer_fn(
#             maxlen=self.maxlen,
#             num_envs=self.runner.num_envs,
#             min_pr=self.min_pr,
#             pr_factor=self._pr_factor,
#             is_factor=self._is_factor,
#             pr_demo=self.pr_demo,
#         )

#     def populate_expert(self, n=None, pct=None, act_fn=None, clean=True):
#         assert (n and not pct) or (pct and not n)
#         num_replays = int(n or pct * self.replay_buffer.maxlen)

#         state_t = self.runner.reset()
#         state_t = self.transform_state(state_t)

#         tqdm.write("Populating Replay Buffer...")
#         for _ in tqdm(range(num_replays)):
#             if act_fn is not None:
#                 action = act_fn(state=U.to_tensor(state_t), step=0)
#             else:
#                 action = self.runner.sample_random_action()
#             sn, reward, done, info = self.runner.act(action)
#             sn = self.transform_state(sn)

#             self.replay_buffer.add_sample_demo(
#                 state=state_t,
#                 action=action,
#                 reward=reward,
#                 done=done,
#                 # info=info,
#             )

#             state_t = sn

#         if clean:
#             self.runner.clean()

import reward.utils as U
from tqdm import tqdm
from reward.batcher import ReplayBatcher
from reward.utils.buffers import DemoReplayBuffer


# TODO: Fix args and kwargs
class DemoReplayBatcher(ReplayBatcher):
    def __init__(self, *args, replay_buffer_fn=DemoReplayBuffer, **kwargs):
        super().__init__(*args, replay_buffer_fn=replay_buffer_fn, **kwargs)

    def _create_replay_buffer(self, replay_buffer_fn):
        return replay_buffer_fn(maxlen=self.maxlen, num_envs=self.runner.num_envs)

    def populate_expert(self, n=None, pct=None, act_fn=None, clean=True):
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
            sn, reward, done, info = self.runner.act(action)
            sn = self.transform_state(sn)

            self.replay_buffer.add_sample_demo(
                state=state_t,
                action=action,
                reward=reward,
                done=done,
                # info=info,
            )

            state_t = sn

        if clean:
            self.runner.clean()
