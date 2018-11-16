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
        stack: int
            Number of sequential states stacked when sampling
        batch_size: int
            Mini-batch size created by sample

    Examples
    --------
        Typical use for atari, with each frame being a 84x84 grayscale
        image (uint8), storing 1M frames should use about 7GiB of RAM
        (8 * 64 * 64 * 1M bits)
    """

    def __init__(self, maxlen, num_envs, stack=1, n_step=1):
        self.maxlen = int(maxlen)
        self.num_envs = num_envs
        # TODO: Real maxlen ?
        self.real_maxlen = self.maxlen // self.num_envs
        self.stack = stack
        self.n_step = n_step
        self.initialized = False
        # Intialized at -1 so the first updated position is 0
        self.idx = -1
        self._len = 0

    def __len__(self):
        return self._len * self.num_envs

    def _get_batch(self, idxs):
        idxs = np.array(idxs)
        # Get states
        sb = self.s_stride[idxs]
        if self.sn is not None:
            snb = self.stp1_stride[idxs]
        else:
            snb = self.s_stride[idxs + self.n_step]
        acs = self.a_stride[idxs, -1:]
        rs = self.r_stride[idxs, -self.n_step :]
        ds = self.d_stride[idxs, -self.n_step :]

        sb = sb.swapaxes(0, 1)
        snb = snb.swapaxes(0, 1)
        acs = acs.swapaxes(0, 1)
        rs = rs.swapaxes(0, 1)
        ds = ds.swapaxes(0, 1)

        return Batch(
            s=sb, sn=snb, ac=acs, r=rs, done=ds, idx=idxs
        )

    @property
    def available_idxs(self):
        return self.num_envs * (len(self) - self.stack - self.n_step + 1)

    def _initialize(self, state, ac, r, done, sn=None):
        self.initialized = True
        maxlen = self.real_maxlen
        # Allocate memory
        self.states = np.empty((maxlen,) + state.shape, dtype=state.dtype)
        self.acs = np.empty((maxlen,) + ac.shape, dtype=ac.dtype)
        self.rs = np.empty((maxlen,) + r.shape, dtype=r.dtype)
        self.ds = np.empty((maxlen,) + done.shape, dtype=np.bool)
        if sn is not None:
            assert state.shape == sn.shape
            self.sn = np.empty((maxlen,) + state.shape, dtype=state.dtype)
        else:
            self.sn = None

        self._create_strides()

    def _create_strides(self):
        # Function for selecting multiple slices
        self.s_stride = strided_axis(arr=self.states, window=self.stack)
        self.a_stride = strided_axis(arr=self.acs, window=self.stack)
        self.r_stride = strided_axis(arr=self.rs, window=self.stack + self.n_step - 1)
        self.d_stride = strided_axis(arr=self.ds, window=self.stack + self.n_step - 1)
        if self.sn is not None:
            self.stp1_stride = strided_axis(arr=self.sn, window=self.stack)
        else:
            self.stp1_stride = strided_axis(arr=self.states, window=self.stack)

    def reset(self):
        self.idx = -1
        self._len = 0

    def add_sample(self, state, ac, r, done, sn=None):
        """
        Add a single sample to the replay buffer.

        Expect transitions to be in the shape of (num_envs, features).
        """
        if not self.initialized:
            self._initialize(
                state=state,
                ac=ac,
                r=r,
                done=done,
                sn=sn,
            )

        self.check_shapes(state, ac, r, done)

        # Update current position
        self.idx = (self.idx + 1) % self.real_maxlen
        self._len = min(self._len + 1, self.real_maxlen)

        # Store transition
        self.states[self.idx] = state
        self.acs[self.idx] = ac
        self.rs[self.idx] = r
        self.ds[self.idx] = done
        if sn is not None:
            assert self.sn is not None
            self.sn[self.idx] = sn

    def add_samples(self, states, acs, rs, ds):
        """
        Add a single sample to the replay buffer.

        Expect transitions to be in the shape of (num_samples, num_envs, features).
        """
        # TODO: Possible optimization using slices
        assert states.shape[0] == acs.shape[0] == rs.shape[0] == ds.shape[0]
        if not self.initialized:
            self._initialize(state=states[0], ac=acs[0], r=rs[0], done=ds[0])
        num_samples = states.shape[0]

        part = range(self.idx + 1, self.idx + 1 + num_samples)
        idxs = np.take(np.arange(self.real_maxlen), part, mode="wrap")

        self.states[idxs] = states
        del states
        self.acs[idxs] = acs
        del acs
        self.rs[idxs] = rs
        del rs
        self.ds[idxs] = ds
        del ds

        # Update current position
        self.idx = (self.idx + num_samples) % self.real_maxlen
        self._len = min(self._len + num_samples, self.real_maxlen)

    def sample(self, batch_size):
        idxs = np.random.choice(self.available_idxs, size=batch_size, replace=False)
        return self._get_batch(idxs=idxs)

    def check_shapes(self, *arrs):
        for arr in arrs:
            dim = arr.shape[0]
            err_msg = "Expect first dimension to be equal num_envs."
            err_msg += " Expected {} but got {}.".format(self.num_envs, dim)
            assert dim == self.num_envs, err_msg

    def save(self, savedir):
        savedir = Path(savedir) / "buffer"
        savedir.mkdir(exist_ok=True)
        tqdm.write("Saving buffer to {}".format(savedir))

        # Save transitions
        np.save(savedir / "states.npy", self.states[: len(self)])
        np.save(savedir / "acs.npy", self.acs[: len(self)])
        np.save(savedir / "rs.npy", self.rs[: len(self)])
        np.save(savedir / "ds.npy", self.ds[: len(self)])

    def load(self, loaddir):
        loaddir = Path(loaddir) / "buffer"
        tqdm.write("Loading buffer from {}".format(loaddir))

        # Load transitions
        states = np.load(loaddir / "states.npy")
        acs = np.load(loaddir / "acs.npy")
        rs = np.load(loaddir / "rs.npy")
        ds = np.load(loaddir / "ds.npy")

        self.add_samples(states=states, acs=acs, rs=rs, ds=ds)


def strided_axis(arr, window):
    """
    https://stackoverflow.com/questions/43413582/selecting-multiple-slices-from-a-numpy-array-at-once/43413801#43413801
    """
    shape = arr.shape
    strides = arr.strides
    num_envs = shape[1]

    num_rolling_windows = num_envs * (shape[0] - window + 1)
    new_shape = (num_rolling_windows, window, *shape[2:])
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
        sn = [self[i + 1]["s"] for i in idxs]
        batch.sn = sn
        batch.idx = idxs
        return batch

    def add_sample(self, state, ac, r, done):
        # If buffer is not full, add a new element
        if len(self.buffer) <= self.maxlen:
            self.buffer.append(None)
        # Store new transition at the appropriate index
        self.position = (self.position + 1) % self.maxlen
        self.buffer[self.position] = dict(
            s=state, ac=ac, r=r, done=done
        )

    def sample(self, batch_size):
        idxs = np.random.choice(len(self) - 1, batch_size, replace=False)
        return self._get_batch(idxs=idxs)

    def save(self, savedir):
        raise NotImplementedError

    def load(self, loaddir):
        raise NotImplementedError
