r"""
A collection of miscellaneous wrappers that can be used on a variety of envs, a bunch of these wrappers have been taken from `OpenAI Gym <https://github.com/openai/gym/tree/master/gym/wrappers>`_.
"""
import numpy as np
from torchrl.envs.wrappers import BaseWrapper


class DelayedStart(BaseWrapper):
    """
    Perform random actions only at the start. Useful for parallel envs ensuring
    every env will be exploring a different part of the state.
    """

    def __init__(self, env, max_delay=1000):
        self.max_delay = max_delay
        self.woke = False
        self.wake_step = np.random.choice(self.max_delay)
        self._step = 0
        super().__init__(env=env)

    def step(self, action):
        if not self.woke:
            action = self.sample_random_action()
        if self._step == self.wake_step:
            self.woke = True
            self.reset()

        self._step += 1
        return self.env.step(action)


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

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.num_lives
        return obs


class RandomReset(BaseWrapper):
    def __init__(self, env, num_actions=30):
        self.num_actions = num_actions
        self.wake_step = np.random.choice(self.num_actions) + 1
        super().__init__(env=env)

    def reset(self):
        self.env.reset()

        for _ in range(self.wake_step):
            action = self.env.sample_random_action()
            state, reward, done, info = self.env.step(action)

            if done:
                state = self.env.reset()

        return state


class FireReset(BaseWrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        assert env.get_action_meanings()[1] == "FIRE"
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
            [2] + list(self.env.get_state_info().shape), dtype=np.uint8
        )
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
