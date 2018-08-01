import numpy as np
import torchrl.utils as U
from torchrl.batchers import BaseBatcher


class RolloutBatcher(BaseBatcher):
    def get_batch(self, select_action_fn):
        super().get_batch(select_action_fn=select_action_fn)
        horizon = self.batch_size // self.runner.num_envs
        batch = U.Batch(
            initial_keys=["state_t_and_tp1", "action", "reward", "done", "info"]
        )

        for i in range(horizon):
            action = select_action_fn(self._state_t, self.num_steps)

            state_tp1, reward, done, info = self.runner.act(action)
            state_tp1 = self.transform_state(state_tp1)

            batch.state_t_and_tp1.append(self._state_t)
            batch.action.append(action)
            batch.reward.append(reward)
            batch.done.append(done)
            batch.info.append(info)

            self._state_t = state_tp1

        batch.state_t_and_tp1.append(self._state_t)
        batch = batch.to_array_or_tensor()

        batch.state_t = batch.state_t_and_tp1[:-1]
        batch.state_tp1 = batch.state_t_and_tp1[1:]

        batch = self.transform_batch(batch)

        return batch

    def transform_state(self, state, training=True):
        """
        Apply functions to state, called before selecting an action.
        """
        state = super().transform_state(state, training=training)
        # TODO
        state = U.to_np(state)
        state = U.to_tensor(state)
        return state


# TODO
# class RolloutBatcher(BaseBatcher):
#     def _allocate_batch(self, num_steps):
#         if self.batch is None or self.steps_per_batch != num_steps:
#             self.steps_per_batch = num_steps
#             state_t_and_tp1 = np.empty(
#                 [num_steps + 1, self.env.num_envs] +
#                 list(self.env.get_state_info().shape),
#                 #                 dtype=self.env.get_state_info().dtype)
#                 dtype=np.float32)
#             action = self._get_action_array()
#             action = np.empty((num_steps, *action.shape), dtype=action.dtype)
#             reward = np.empty([num_steps] + [self.env.num_envs], dtype=np.float32)
#             done = np.empty([num_steps] + [self.env.num_envs])

#             self.batch = U.Batch(
#                 state_t_and_tp1=state_t_and_tp1, action=action, reward=reward, done=done)

#     def _get_action_array(self):
#         action_info = self.env.get_action_info()
#         if action_info.space == 'continuous':
#             shape = (self.env.num_envs, np.prod(action_info.shape))
#             # action_type = np.float32
#         elif action_info.space == 'discrete':
#             shape = (self.env.num_envs, )
#             # action_type = np.int32
#         else:
#             raise ValueError('Action dtype {} not implemented'.format(action_info.dtype))
#         return np.zeros(shape, dtype=action_info.dtype)

#     def get_batch(self, select_action_fn):
#         super().get_batch(select_action_fn=select_action_fn)
#         horizon = self.batch_size // self.env.num_envs
#         self._allocate_batch(horizon)

#         for i in range(horizon):
#             action = select_action_fn(self._state_t)

#             state_tp1, reward, done, info = self.env.step(action)
#             state_tp1 = self.transform_state(state_tp1)

#             self.batch.state_t_and_tp1[i] = self._state_t
#             self.batch.action[i] = action
#             self.batch.reward[i] = reward
#             self.batch.done[i] = done
#             self._state_t = state_tp1

#         self.batch.state_t_and_tp1[i + 1] = state_tp1
#         self.batch.state_t = self.batch.state_t_and_tp1[:-1]
#         self.batch.state_tp1 = self.batch.state_t_and_tp1[1:]

#         return self.transform_batch(self.batch)
