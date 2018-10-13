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
        self.maxlen = maxlen
        self.num_envs = num_envs
        self.real_maxlen = maxlen // num_envs
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

        # Concatenate first two dimensions (num_envs, num_samples)
        #         b_states_t = join_first_dims(b_states_t, num_dims=2)
        #         b_states_tp1 = join_first_dims(b_states_tp1, num_dims=2)
        #         actions = join_first_dims(actions, num_dims=2)
        #         rewards = join_first_dims(rewards, num_dims=2)
        #         dones = join_first_dims(dones, num_dims=2)
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

        idxs = range(self.current_idx, self.current_idx + states.shape[0])
        self.states.put(idxs, states, mode="wrap")
        self.actions.put(idxs, actions, mode="wrap")
        self.rewards.put(idxs, rewards, mode="wrap")
        self.dones.put(idxs, dones, mode="wrap")

        # Update current position
        self.current_idx = (self.current_idx + states.shape[0]) % self.real_maxlen
        self.current_len = min(self.current_len + states.shape[0], self.real_maxlen)

    def sample(self, batch_size):
        idxs = np.random.randint(self.available_idxs, size=batch_size)
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

        # Save hyperparameters
        ignore = [
            "states",
            "actions",
            "rewards",
            "dones",
            "s_stride",
            "a_stride",
            "r_stride",
            "d_stride",
        ]
        d = {k: v for k, v in self.__dict__.items() if k not in ignore}
        with open(str(savedir / "params.json"), "w") as f:
            json.dump(d, f, indent=4)

    def load(self, loaddir):
        # TODO: This overwrites all other paramters, maybe this should be a class method
        loaddir = Path(loaddir) / "buffer"
        tqdm.write("Loading buffer from {}".format(loaddir))

        with open(loaddir / "params.json") as f:
            params = json.load(f)
        self.__dict__.update(params)

        # Load transitions
        self.states = np.load(loaddir / "states.npy")
        self.actions = np.load(loaddir / "actions.npy")
        self.rewards = np.load(loaddir / "rewards.npy")
        self.dones = np.load(loaddir / "dones.npy")

        self._create_strides()
        self.initialized = True


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
