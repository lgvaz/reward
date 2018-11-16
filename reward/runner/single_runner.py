import numpy as np
from reward.runner import BaseRunner
from boltons.cacheutils import cachedproperty


class SingleRunner(BaseRunner):
    def __init__(self, env, ep_maxlen=None):
        super().__init__(env=env, ep_maxlen=ep_maxlen)
        self._ep_reward_sum = 0
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
    def action_space(self):
        return self.env.action_space

    def reset(self):
        self._ep_reward_sum = 0
        self._ep_num_steps = 0

        state = self.env.reset()
        return state[None]

    def act(self, action):
        # TODO: Squeezing action may break some cases (when action is not an array)
        # Pendulum-v0 was not working correctly if action were not squeezed
        state, reward, done, info = self.env.step(action)
        state = state[None]

        self._ep_reward_sum += reward
        self.num_steps += 1
        self._ep_num_steps += 1
        if done or self._ep_num_steps >= self.ep_maxlen:
            self.rs.append(self._ep_reward_sum)
            self.ep_lens.append(self._ep_num_steps)
            state = self.reset()

        return state, np.array(reward)[None], np.array(done)[None], info

    def sample_random_action(self):
        return np.array(self.env.sample_random_action())[None]

    def close(self):
        self.env.close()
