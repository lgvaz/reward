# import numpy as np
# from torchrl.utils import join_first_dims, Batch

# class ReplayBuffer:
#     '''
#     Memory efficient implementation of replay buffer, storing each state only once.

#     Parameters
#     ----------
#         maxlen: int
#             Maximum number of transitions stored
#         history_len: int
#             Number of sequential states stacked when sampling
#         batch_size: int
#             Mini-batch size created by sample

#     Examples
#     --------
#         Typical use for atari, with each frame being a 84x84 grayscale
#         image (uint8), storing 1M frames should use about 7GiB of RAM
#         (8 * 64 * 64 * 1M bits)
#     '''

#     def __init__(self, maxlen, num_envs, history_len=1, batch_size=32, n_step=1):
#         self.maxlen = maxlen
#         self.num_envs = num_envs
#         self.real_maxlen = maxlen // num_envs
#         self.history_len = history_len
#         self.batch_size = batch_size
#         self.n_step = n_step
#         self.initialized = False
#         self.current_idx = 0
#         self.current_len = 0

#     def __len__(self):
#         return self.maxlen

#     def add(self, state, action, reward, done):
#         self.check_shapes(state, action, reward, done)
#         if not self.initialized:
#             self.initialized = True
#             # TODO: Squeeze?
#             #             state_shape = np.squeeze(state).shape
#             # Allocate memory
#             # TODO: state dtype
#             self.states = np.empty((self.real_maxlen, ) + state.shape, dtype=state.dtype)
#             self.actions = np.empty(
#                 (self.real_maxlen, ) + action.shape, dtype=action.dtype)
#             self.rewards = np.empty(
#                 (self.real_maxlen, ) + reward.shape, dtype=reward.dtype)
#             self.dones = np.empty((self.real_maxlen, ) + done.shape, dtype=np.bool)

#             # Function for selecting multiple slices
#             self.states_stride_history = strided_axis(
#                 arr=self.states, window_size=self.history_len)
#             self.actions_stride = strided_axis(
#                 arr=self.actions, window_size=self.history_len)
#             self.rewards_stride_nstep = strided_axis(
#                 arr=self.rewards, window_size=self.history_len + self.n_step - 1)
#             self.dones_stride_nstep = strided_axis(
#                 arr=self.dones, window_size=self.history_len + self.n_step - 1)

#         # Store transition
#         # TODO: Squeeze?
#         self.states[self.current_idx] = state
#         self.actions[self.current_idx] = action
#         self.rewards[self.current_idx] = reward
#         self.dones[self.current_idx] = done

#         # Update current position
#         self.current_idx = (self.current_idx + 1) % self.real_maxlen
#         self.current_len = min(self.current_len + 1, self.real_maxlen)

#     def sample(self):
#         idxs = self.generate_idxs(n=self.batch_size)
#         # Get states
#         b_states_t = self.states_stride_history[idxs]
#         b_states_tp1 = self.states_stride_history[idxs + self.n_step]
#         actions = self.actions_stride[idxs, -1:]
#         rewards = self.rewards_stride_nstep[idxs, -self.n_step:]
#         dones = self.dones_stride_nstep[idxs, -self.n_step:]

#         # Concatenate first two dimensions (num_samples, num_envs)
#         #         b_states_t = join_first_dims(b_states_t, num_dims=2)
#         #         b_states_tp1 = join_first_dims(b_states_tp1, num_dims=2)
#         #         actions = join_first_dims(actions, num_dims=2)
#         #         rewards = join_first_dims(rewards, num_dims=2)
#         #         dones = join_first_dims(dones, num_dims=2)

#         return Batch(
#             state_t=b_states_t,
#             state_tp1=b_states_tp1,
#             action=actions,
#             reward=rewards,
#             done=dones)

#     def generate_idxs(self, n):
#         return np.random.randint(
#             self.num_envs * (self.current_len - self.history_len - self.n_step + 1),
#             size=n)
# #         end_idxs = start_idxs + self.history_len - 1

#     def check_shapes(self, *arrs):
#         for arr in arrs:
#             assert arr.shape[0] == self.num_envs

# # def strided_axis(arr, window):
# #     '''
# #     https://stackoverflow.com/questions/43413582/selecting-multiple-slices-from-a-numpy-array-at-once/43413801#43413801
# #     '''
# #     shape = arr.shape
# #     strides = arr.strides

# #     new_len = shape[0] - window + 1
# #     new_shape = (new_len, shape[1], window, *shape[2:])
# #     new_strides = (strides[0], strides[1], strides[0], *strides[2:])

# #     return np.lib.stride_tricks.as_strided(
# #         arr, shape=new_shape, strides=new_strides, writeable=False)

# def strided_axis(arr, window_size):
#     '''
#     https://stackoverflow.com/questions/43413582/selecting-multiple-slices-from-a-numpy-array-at-once/43413801#43413801
#     '''
#     shape = arr.shape
#     strides = arr.strides
#     num_envs = shape[1]

#     num_rolling_windows = num_envs * (shape[0] - window_size + 1)
#     new_shape = (num_rolling_windows, window_size, *shape[2:])
#     new_strides = (strides[1], num_envs * strides[1], *strides[2:])

#     return np.lib.stride_tricks.as_strided(
#         arr, shape=new_shape, strides=new_strides, writeable=False)
import random
from torchrl.utils.memories import SimpleMemory
from torchrl.utils import Batch
from collections import deque


class ReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return list(self.buffer)[key]

    def add_sample(self, **kwargs):
        # Dict of lists to list of dicts
        sample = [SimpleMemory(zip(kwargs, v)) for v in zip(*kwargs.values())]
        self.buffer.extend(sample)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, k=batch_size)
        return Batch.from_list_of_dicts(samples)
