import numpy as np
import torchrl.utils as U
from torchrl.batchers import BaseBatcher
from tqdm import tqdm


class ReplayBatcher(BaseBatcher):
    def __init__(
        self,
        runner,
        batch_size,
        *,
        learning_freq=1,
        transforms=None,
        replay_buffer=None,
        replay_buffer_maxlen=1e6,
        init_replays=0.05,
    ):
        super().__init__(runner=runner, batch_size=batch_size, transforms=transforms)
        self.learning_freq = learning_freq
        self.replay_buffer = replay_buffer or U.buffers.ReplayBuffer(
            maxlen=int(replay_buffer_maxlen), num_envs=self.runner.num_envs
        )
        self.init_replays = init_replays

        self.populate_replay_buffer()

    def populate_replay_buffer(self):
        state_t = self.runner.reset()
        state_t = self.transform_state(state_t)

        tqdm.write("Populating Replay Buffer...")
        for _ in tqdm(range(int(self.init_replays * self.replay_buffer.maxlen))):
            action = self.runner.sample_random_action()
            state_tp1, reward, done, info = self.runner.act(action)
            state_tp1 = self.transform_state(state_tp1)

            self.replay_buffer.add_sample(
                state_t=state_t,
                state_tp1=state_tp1,
                action=action,
                reward=reward,
                done=done,
                # info=info,
            )

            state_t = state_tp1

    def get_batch(self, select_action_fn):
        super().get_batch(select_action_fn=select_action_fn)

        for i in range(self.learning_freq):
            # TODO: Maybe the array can live in pinned memory?
            state_t = U.to_tensor(U.to_np(self.state_t))
            action = select_action_fn(state_t, self.num_steps)

            state_tp1, reward, done, info = self.runner.act(action)
            state_tp1 = self.transform_state(state_tp1)

            self.replay_buffer.add_sample(
                state_t=self.state_t,
                state_tp1=state_tp1,
                action=action,
                reward=reward,
                done=done,
                # info=info,
            )

            self.state_t = state_tp1

        batch = self.replay_buffer.sample(self.batch_size)
        # TODO: Maybe to_tensor states here
        return batch
