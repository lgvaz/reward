r"""
A collection of miscellaneous wrappers that can be used on a variety of envs, a bunch of these wrappers have been taken from `OpenAI Gym <https://github.com/openai/gym/tree/master/gym/wrappers>`_.
"""
import numpy as np
from reward.env.wrappers import BaseWrapper


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

    def step(self, ac):
        if not self.woke:
            ac = self.sample_random_ac()
        if self._step == self.wake_step:
            self.woke = True
            self.reset()

        self._step += 1
        return self.env.step(ac)


class EpisodicLife(BaseWrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        self.lives = 0
        self.was_real_d = True
        super().__init__(env=env)

    def step(self, ac):
        obs, r, d, info = self.env.step(ac)
        self.was_real_d = d
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.num_lives
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises d.
            d = True
        self.lives = lives
        return obs, r, d, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_d:
            obs = self.env.reset()
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.num_lives
        return obs


class RandomReset(BaseWrapper):
    def __init__(self, env, num_acs=30):
        self.num_acs = num_acs
        self.wake_step = np.random.choice(self.num_acs) + 1
        super().__init__(env=env)

    def reset(self):
        self.env.reset()

        for _ in range(self.wake_step):
            ac = self.env.sample_random_ac()
            s, r, d, info = self.env.step(ac)

            if d:
                s = self.env.reset()

        return s


class FireReset(BaseWrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        assert env.get_ac_meanings()[1] == "FIRE"
        assert len(env.get_ac_meanings()) >= 3
        super().__init__(env=env)

    def reset(self):
        s = self.env.reset()
        s, r, d, _ = self.env.step(1)
        if d:
            self.env.reset()
        s, r, d, _ = self.env.step(2)
        if d:
            self.env.reset()

        return s


class ActionRepeat(BaseWrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env=env)
        self._obs_buffer = np.zeros([2] + list(self.env.s_space.shape), dtype=np.uint8)
        self._skip = skip

    def step(self, ac):
        """Repeat action, sum reward, and max over last observations."""
        total_r = 0.0
        d = None
        for i in range(self._skip):
            obs, r, d, info = self.env.step(ac)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_r += r
            if d:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_r, d, info


class ActionBound(BaseWrapper):
    def __init__(self, env, low=-1, high=1):
        super().__init__(env=env)
        self.low = low
        self.high = high
        # TODO: Maybe assert continuous?
        self.mapper = self._map_range(
            old_low=self.low,
            old_high=self.high,
            new_low=self.ac_space.low,
            new_high=self.ac_space.high,
        )
        self.mapper_inverse = self._map_range(
            old_low=self.ac_space.low,
            old_high=self.ac_space.high,
            new_low=self.low,
            new_high=self.high,
        )

    def _map_range(self, old_low, old_high, new_low, new_high):
        old_span = old_high - old_low
        new_span = new_high - new_low

        def get(value):
            norm_value = (value - old_low) / old_span
            return new_low + (norm_value * new_span)

        return get

    def step(self, ac):
        ac = self.mapper(ac)
        return self.env.step(ac)

    def sample_random_ac(self):
        ac = self.env.sample_random_ac()
        return self.mapper_inverse(ac)
