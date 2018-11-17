import pdb
import numpy as np
import reward.utils as U
from reward.batcher import BaseBatcher
from tqdm.autonotebook import tqdm
from reward.batcher.transforms import StackStates
from reward.utils.buffers import ReplayBuffer


# TODO: Renames: leraning_freq and grad_steps_per_batch
class ReplayBatcher(BaseBatcher):
    def __init__(
        self,
        runner,
        batch_size,
        *,
        learning_freq=1,
        grad_steps_per_batch=1,
        transforms=None,
        replay_buffer_fn=ReplayBuffer,
        maxlen=1e6,
        init_replays=0.05,
    ):
        super().__init__(runner=runner, batch_size=batch_size, transforms=transforms)
        self._check_transforms()
        self.maxlen = maxlen
        self.learning_freq = learning_freq
        self.grad_steps_per_batch = grad_steps_per_batch
        self.replay_buffer = self._create_replay_buffer(replay_buffer_fn)
        self._grad_iter = 0
        self.init_replays = init_replays

    def _create_replay_buffer(self, replay_buffer_fn):
        return replay_buffer_fn(maxlen=self.maxlen, num_envs=self.runner.num_envs)

    def _check_transforms(self):
        # TODO: Hack for handling StackStates transform together with replay_buffer
        # TODO: Add support for StackStates dim != 1
        # TODO TODO TODO: Modify replay buffer history length
        self.s_stacker = None
        self.n_stacks = 1
        for tfm in self.transforms.copy():
            if isinstance(tfm, StackStates):
                self.n_stacks = tfm.n
                self.s_stacker = tfm
                transforms.remove(tfm)

    def populate(self, n=None, pct=None, act_fn=None, clean=True):
        assert (n and not pct) or (pct and not n)
        num_replays = int(n or pct * self.replay_buffer.maxlen)

        s = self.runner.reset()
        # s = self.transform_s(s)

        tqdm.write("Populating Replay Buffer...")
        for _ in tqdm(range(num_replays)):
            if act_fn is not None:
                s_tfm = self.transform_s(self.s)
                if self.s_stacker is not None:
                    s_tfm = self.s_stacker.transform_s(s_tfm)
                ac = act_fn(s=U.to_tensor(s_tfm), step=0)
            else:
                ac = self.runner.sample_random_ac()
            sn, r, d, info = self.runner.act(ac)
            # sn = self.transform_s(sn)

            self.replay_buffer.add_sample(
                s=s,
                # TODO: sn here only for testing
                sn=sn,
                ac=ac,
                r=r,
                d=d,
                # info=info,
            )

            s = sn

        if clean:
            self.runner.clean()

    def get_batch(self, act_fn):
        self._grad_iter = (self._grad_iter + 1) % self.grad_steps_per_batch
        if self._grad_iter == 0:
            for i in range(self.learning_freq):
                if self.s is None:
                    self.s = self.runner.reset()
                s_tfm = self.transform_s(self.s)
                # TODO: Hacky way of stacking
                if self.s_stacker is not None:
                    s_tfm = self.s_stacker.transform_s(s_tfm)
                ac = act_fn(U.to_tensor(s_tfm), self.num_steps)

                sn, r, d, info = self.runner.act(ac)

                self.replay_buffer.add_sample(
                    s=self.s,
                    # TODO: sn here only for testing
                    sn=sn,
                    ac=ac,
                    r=r,
                    d=d,
                    # info=info,
                )

                self.s = sn

        batch = self.replay_buffer.sample(self.batch_size)
        # TODO: Refactor next lines, training=False incorrect?
        batch.s = self.transform_s(
            U.join_first_dims(batch.s, 2), training=False
        ).reshape(batch.s.shape)
        batch.sn = self.transform_s(
            U.join_first_dims(batch.sn, 2), training=False
        ).reshape(batch.sn.shape)
        batch = self.transform_batch(batch)
        # TODO: Check if this next lines are correct
        batch.r = batch.r[..., None]
        batch.d = batch.d[..., None]
        # TODO: Maybe to_tensor states here
        return batch

    def reset(self):
        self.replay_buffer.reset()

    def save_exp(self, savedir):
        self.replay_buffer.save(savedir=savedir)

    def load_exp(self, loaddir):
        self.replay_buffer.load(loaddir=loaddir)
