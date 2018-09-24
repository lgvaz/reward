import pdb
import numpy as np
import reward.utils as U
from reward.batchers import BaseBatcher
from tqdm.autonotebook import tqdm
from reward.batchers.transforms import StackStates


# TODO: replay_buffer_maxlen -> maxlen
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
        replay_buffer_maxlen=1e6,
        init_replays=0.05,
    ):
        self._check_transforms(transforms)
        super().__init__(runner=runner, batch_size=batch_size, transforms=transforms)
        self.learning_freq = learning_freq
        self.grad_steps_per_batch = grad_steps_per_batch
        self.replay_buffer = self._create_replay_buffer(int(replay_buffer_maxlen))
        self._grad_iter = 0
        self.init_replays = init_replays

    def _create_replay_buffer(self, maxlen):
        return U.buffers.ReplayBuffer(maxlen=maxlen, num_envs=self.runner.num_envs)

    def _check_transforms(self, transforms):
        # TODO: Hack for handling StackStates transform together with replay_buffer
        # TODO: Add support for StackStates dim != 1
        self.state_stacker = None
        self.n_stacks = 1
        for tfm in transforms.copy():
            if isinstance(tfm, StackStates):
                self.n_stacks = tfm.n
                self.state_stacker = tfm
                transforms.remove(tfm)

    def populate(self, n=None, pct=None, get_action_fn=None):
        assert (n and not pct) or (pct and not n)
        num_replays = int(n or pct * self.replay_buffer.maxlen)

        state_t = self.runner.reset()
        state_t = self.transform_state(state_t)

        tqdm.write("Populating Replay Buffer...")
        for _ in tqdm(range(num_replays)):
            if get_action_fn is not None:
                action = get_action_fn(state=U.to_tensor(state_t), step=0)
            else:
                action = self.runner.sample_random_action()
            state_tp1, reward, done, info = self.runner.act(action)
            state_tp1 = self.transform_state(state_tp1)

            self.replay_buffer.add_sample(
                state=state_t,
                # state_t=state_t,
                # state_tp1=state_tp1,
                action=action,
                reward=reward,
                done=done,
                # info=info,
            )

            state_t = state_tp1

    def get_batch(self, get_action_fn):
        if self.state_t is None:
            self.state_t = self.runner.reset()

        self._grad_iter = (self._grad_iter + 1) % self.grad_steps_per_batch
        if self._grad_iter == 0:
            for i in range(self.learning_freq):
                state_t_tfm = self.transform_state(self.state_t)
                # TODO: Hacky way of stacking
                if self.state_stacker is not None:
                    state_t_tfm = self.state_stacker.transform_state(state_t_tfm)
                action = get_action_fn(U.to_tensor(state_t_tfm), self.num_steps)

                state_tp1, reward, done, info = self.runner.act(action)
                # state_tp1 = self.transform_state(state_tp1)

                self.replay_buffer.add_sample(
                    state=self.state_t,
                    # state_t=self.state_t,
                    # state_tp1=state_tp1,
                    action=action,
                    reward=reward,
                    done=done,
                    # info=info,
                )

                self.state_t = state_tp1

        batch = self.replay_buffer.sample(self.batch_size)
        # TODO: Refactor next lines, training=False incorrect?
        batch.state_t = self.transform_state(
            U.join_first_dims(batch.state_t, 2), training=False
        ).reshape(batch.state_t.shape)
        batch.state_tp1 = self.transform_state(
            U.join_first_dims(batch.state_tp1, 2), training=False
        ).reshape(batch.state_tp1.shape)
        batch = self.transform_batch(batch)
        # TODO: Check if this next lines are correct
        batch.reward = batch.reward[..., None]
        batch.done = batch.done[..., None]
        # TODO: Maybe to_tensor states here
        return batch
