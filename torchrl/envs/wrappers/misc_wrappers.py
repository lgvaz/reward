r'''
A collection of miscellaneous wrappers that can be used on a variety of envs, a bunch of these wrappers have been taken from `OpenAI Gym <https://github.com/openai/gym/tree/master/gym/wrappers>`_.
'''
import numpy as np
from torchrl.envs.wrappers import BaseWrapper


class EpisodicLife(BaseWrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        self.lives = 0
        self.was_real_done = True
        super().__init__(env=env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.num_lives
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info


class RandomReset(BaseWrapper):
    def __init__(self, env, num_actions=30):
        self.num_actions = num_actions
        super().__init__(env=env)

    def reset(self):
        self.env.reset()

        for _ in range(self.num_actions):
            action = self.env.sample_random_action()
            state, reward, done, info = self.env.step(action)

            if done:
                state = self.env.reset()

        return state


class FireReset(BaseWrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        assert env.get_action_meanings()[1] == 'FIRE'
        assert len(env.get_action_meanings()) >= 3
        super().__init__(env=env)

    def reset(self):
        state = self.env.reset()
        state, reward, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        state, reward, done, _ = self.env.step(2)
        if done:
            self.env.reset()

        return state


class ActionRepeat(BaseWrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env=env)
        self._obs_buffer = np.zeros(
            [2] + list(self.env.get_state_info().shape), dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info


class HWC_to_CHW(BaseWrapper):
    def reset(self):
        state = self.env.reset()
        assert state.ndim == 3, 'state have {} dims and must have 3'.format(state.ndim)
        state = np.rollaxis(state, -1)

        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        assert state.ndim == 3, 'state have {} dims and must have 3'.format(state.ndim)
        state = np.rollaxis(state, -1)

        return state, reward, done, info
