import pdb
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
from reward.utils import Batch, to_np


class ReplayBuffer:
    """
    Memory efficient implementation of replay buffer, storing each state only once.

    Parameters
    ----------
        maxlen: int
            Maximum number of transitions stored
        history_len: int
            Number of sequential states stacked when sampling
        batch_size: int
            Mini-batch size created by sample

    Examples
    --------
        Typical use for atari, with each frame being a 84x84 grayscale
        image (uint8), storing 1M frames should use about 7GiB of RAM
        (8 * 64 * 64 * 1M bits)
    """

    def __init__(self, maxlen, num_envs, history_len=1, n_step=1):
        self.maxlen = int(maxlen)
        self.num_envs = num_envs
        self.real_maxlen = self.maxlen // self.num_envs
        self.history_len = history_len
        self.n_step = n_step
        self.initialized = False
        #         # Intialized at -1 so the first updated position is 0
        self.current_idx = -1
        self.current_len = 0

    def __len__(self):
        return self.current_len * self.num_envs

    def _get_batch(self, idxs):
        idxs = np.array(idxs)
        # Get states
        b_states_t = self.s_stride[idxs]
        b_states_tp1 = self.s_stride[idxs + self.n_step]
        actions = self.a_stride[idxs, -1:]
        rewards = self.r_stride[idxs, -self.n_step :]
        dones = self.d_stride[idxs, -self.n_step :]

        b_states_t = b_states_t.swapaxes(0, 1)
        b_states_tp1 = b_states_tp1.swapaxes(0, 1)
        actions = actions.swapaxes(0, 1)
        rewards = rewards.swapaxes(0, 1)
        dones = dones.swapaxes(0, 1)

        return Batch(
            state_t=b_states_t,
            state_tp1=b_states_tp1,
            action=actions,
            reward=rewards,
            done=dones,
            idx=idxs,
        )

    @property
    def available_idxs(self):
        return self.num_envs * (len(self) - self.history_len - self.n_step + 1)

    def _initialize(self, state, action, reward, done):
        self.initialized = True
        # Allocate memory
        self.states = np.empty((self.real_maxlen,) + state.shape, dtype=state.dtype)
        self.actions = np.empty((self.real_maxlen,) + action.shape, dtype=action.dtype)
        self.rewards = np.empty((self.real_maxlen,) + reward.shape, dtype=reward.dtype)
        self.dones = np.empty((self.real_maxlen,) + done.shape, dtype=np.bool)

        self._create_strides()

    def _create_strides(self):
        # Function for selecting multiple slices
        self.s_stride = strided_axis(arr=self.states, window_size=self.history_len)
        self.a_stride = strided_axis(arr=self.actions, window_size=self.history_len)
        self.r_stride = strided_axis(
            arr=self.rewards, window_size=self.history_len + self.n_step - 1
        )
        self.d_stride = strided_axis(
            arr=self.dones, window_size=self.history_len + self.n_step - 1
        )

    def reset(self):
        self.current_idx = -1
        self.current_len = 0

    def add_sample(self, state, action, reward, done):
        """
        Add a single sample to the replay buffer.

        Expect transitions to be in the shape of (num_envs, features).
        """
        if not self.initialized:
            self._initialize(state=state, action=action, reward=reward, done=done)

        self.check_shapes(state, action, reward, done)

        # Update current position
        self.current_idx = (self.current_idx + 1) % self.real_maxlen
        self.current_len = min(self.current_len + 1, self.real_maxlen)

        # Store transition
        self.states[self.current_idx] = state
        self.actions[self.current_idx] = action
        self.rewards[self.current_idx] = reward
        self.dones[self.current_idx] = done

    def add_samples(self, states, actions, rewards, dones):
        """
        Add a single sample to the replay buffer.

        Expect transitions to be in the shape of (num_samples, num_envs, features).
        """
        # TODO: Possible optimization using slices
        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == dones.shape[0]
        if not self.initialized:
            self._initialize(
                state=states[0], action=actions[0], reward=rewards[0], done=dones[0]
            )
        num_samples = states.shape[0]

        part = range(self.current_idx + 1, self.current_idx + 1 + num_samples)
        idxs = np.take(np.arange(self.real_maxlen), part, mode="wrap")

        self.states[idxs] = states
        del states
        self.actions[idxs] = actions
        del actions
        self.rewards[idxs] = rewards
        del rewards
        self.dones[idxs] = dones
        del dones

        # Update current position
        self.current_idx = (self.current_idx + num_samples) % self.real_maxlen
        self.current_len = min(self.current_len + num_samples, self.real_maxlen)

    def sample(self, batch_size):
        idxs = np.random.choice(self.available_idxs, size=batch_size, replace=False)
        return self._get_batch(idxs=idxs)

    def check_shapes(self, *arrs):
        for arr in arrs:
            assert arr.shape[0] == self.num_envs

    def save(self, savedir):
        savedir = Path(savedir) / "buffer"
        savedir.mkdir(exist_ok=True)
        tqdm.write("Saving buffer to {}".format(savedir))

        # Save transitions
        np.save(savedir / "states.npy", self.states[: len(self)])
        np.save(savedir / "actions.npy", self.actions[: len(self)])
        np.save(savedir / "rewards.npy", self.rewards[: len(self)])
        np.save(savedir / "dones.npy", self.dones[: len(self)])

    def load(self, loaddir):
        loaddir = Path(loaddir) / "buffer"
        tqdm.write("Loading buffer from {}".format(loaddir))

        # Load transitions
        states = np.load(loaddir / "states.npy")
        actions = np.load(loaddir / "actions.npy")
        rewards = np.load(loaddir / "rewards.npy")
        dones = np.load(loaddir / "dones.npy")

        self.add_samples(states=states, actions=actions, rewards=rewards, dones=dones)


def strided_axis(arr, window_size):
    """
    https://stackoverflow.com/questions/43413582/selecting-multiple-slices-from-a-numpy-array-at-once/43413801#43413801
    """
    shape = arr.shape
    strides = arr.strides
    num_envs = shape[1]

    num_rolling_windows = num_envs * (shape[0] - window_size + 1)
    new_shape = (num_rolling_windows, window_size, *shape[2:])
    new_strides = (strides[1], num_envs * strides[1], *strides[2:])

    return np.lib.stride_tricks.as_strided(
        arr, shape=new_shape, strides=new_strides, writeable=False
    )


class DictReplayBuffer:
    # TODO: Save and load
    def __init__(self, maxlen, num_envs):
        assert num_envs == 1
        self.maxlen = int(maxlen)
        self.buffer = []
        # Intialized at -1 so the first updated position is 0
        self.position = -1

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return self.buffer[key]

    def _get_batch(self, idxs):
        samples = [self[i] for i in idxs]
        batch = Batch.from_list_of_dicts(samples)
        # Add next state to batch
        state_tp1 = [self[i + 1]["state_t"] for i in idxs]
        batch.state_tp1 = state_tp1
        batch.idx = idxs
        return batch

    def add_sample(self, state, action, reward, done):
        # If buffer is not full, add a new element
        if len(self.buffer) <= self.maxlen:
            self.buffer.append(None)
        # Store new transition at the appropriate index
        self.position = (self.position + 1) % self.maxlen
        self.buffer[self.position] = dict(
            state_t=state, action=action, reward=reward, done=done
        )

    def sample(self, batch_size):
        idxs = np.random.choice(len(self) - 1, batch_size, replace=False)
        return self._get_batch(idxs=idxs)

    def save(self, savedir):
        raise NotImplementedError

    def load(self, loaddir):
        raise NotImplementedError
