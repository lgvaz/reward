import numpy as np
from reward.runner import BaseRunner
from boltons.cacheutils import cachedproperty


class SingleRunner(BaseRunner):
    def __init__(self, env, ep_maxlen=None):
        super().__init__(env=env, ep_maxlen=ep_maxlen)
        self._ep_r_sum = 0
        self._ep_num_steps = 0

    @property
    def env_name(self):
        return self.env.env_name

    @property
    def num_envs(self):
        return 1

    @cachedproperty
    def state_space(self):
        space = self.env.state_space
        space.shape = (1,) + space.shape
        return space

    @cachedproperty
    def ac_space(self):
        return self.env.ac_space

    def reset(self):
        self._ep_r_sum = 0
        self._ep_num_steps = 0

        state = self.env.reset()
        return state[None]

    def act(self, ac):
        # TODO: Squeezing action may break some cases (when action is not an array)
        # Pendulum-v0 was not working correctly if action were not squeezed
        state, r, d, info = self.env.step(ac)
        state = state[None]

        self._ep_r_sum += r
        self.num_steps += 1
        self._ep_num_steps += 1
        if d or self._ep_num_steps >= self.ep_maxlen:
            self.rs.append(self._ep_r_sum)
            self.ep_lens.append(self._ep_num_steps)
            state = self.reset()

        return state, np.array(r)[None], np.array(d)[None], info

    def sample_random_ac(self):
        return np.array(self.env.sample_random_ac())[None]

    def close(self):
        self.env.close()
