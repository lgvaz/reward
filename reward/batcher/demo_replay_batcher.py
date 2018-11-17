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
#         rbuff_fn=DemoReplayBuffer,
#         **kwargs
#     ):
#         self.pr_demo = pr_demo
#         super().__init__(
#             *args,
#             min_pr=min_pr,
#             pr_factor=pr_factor,
#             is_factor=is_factor,
#             rbuff_fn=rbuff_fn,
#             **kwargs
#         )

#     def _create_rbuff(self, rbuff_fn):
#         return rbuff_fn(
#             maxlen=self.maxlen,
#             num_envs=self.runner.num_envs,
#             min_pr=self.min_pr,
#             pr_factor=self._pr_factor,
#             is_factor=self._is_factor,
#             pr_demo=self.pr_demo,
#         )

#     def populate_expert(self, n=None, pct=None, act_fn=None, clean=True):
#         assert (n and not pct) or (pct and not n)
#         num_replays = int(n or pct * self.rbuff.maxlen)

#         s = self.runner.reset()
#         s = self.transform_s(s)

#         tqdm.write("Populating Replay Buffer...")
#         for _ in tqdm(range(num_replays)):
#             if act_fn is not None:
#                 ac = act_fn(s=U.to_tensor(s), step=0)
#             else:
#                 ac = self.runner.sample_random_ac()
#             sn, reward, d, info = self.runner.act(ac)
#             sn = self.transform_s(sn)

#             self.rbuff.add_sample_demo(
#                 s=s,
#                 ac=ac,
#                 r=r,
#                 d=d,
#                 # info=info,
#             )

#             s = sn

#         if clean:
#             self.runner.clean()

import reward.utils as U
from tqdm import tqdm
from reward.batcher import ReplayBatcher
from reward.utils.buffers import DemoReplayBuffer


# TODO: Fix args and kwargs
class DemoReplayBatcher(ReplayBatcher):
    def __init__(self, *args, rbuff_fn=DemoReplayBuffer, **kwargs):
        super().__init__(*args, rbuff_fn=rbuff_fn, **kwargs)

    def _create_rbuff(self, rbuff_fn):
        return rbuff_fn(maxlen=self.maxlen, num_envs=self.runner.num_envs)

    def populate_expert(self, n=None, pct=None, act_fn=None, clean=True):
        assert (n and not pct) or (pct and not n)
        num_replays = int(n or pct * self.rbuff.maxlen)

        s = self.runner.reset()
        s = self.transform_s(s)

        tqdm.write("Populating Replay Buffer...")
        for _ in tqdm(range(num_replays)):
            if act_fn is not None:
                ac = act_fn(s=U.to_tensor(s), step=0)
            else:
                ac = self.runner.sample_random_ac()
            sn, r, d, info = self.runner.act(ac)
            sn = self.transform_s(sn)

            self.rbuff.add_sample_demo(
                s=s,
                ac=ac,
                r=r,
                d=d,
                # info=info,
            )

            s = sn

        if clean:
            self.runner.clean()
