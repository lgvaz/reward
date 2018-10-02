import numpy as np
from reward.runner import BaseRunner


class SingleRunner(BaseRunner):
    def __init__(self, env):
        super().__init__(env=env)
        self._ep_reward_sum = 0
        self._ep_num_steps = 0

    @property
    def env_name(self):
        return self.env.env_name

    @property
    def num_envs(self):
        return 1

    def reset(self):
        state = self.env.reset()
        return state[None]

    def act(self, action):
        # TODO: Squeezing action may break some cases (when action is not an array)
        # Pendulum-v0 was not working correctly if action were not squeezed
        state, reward, done, info = self.env.step(action)

        self._ep_reward_sum += reward
        self.num_steps += 1
        self._ep_num_steps += 1
        if done:
            self.rewards.append(self._ep_reward_sum)
            self.ep_lens.append(self._ep_num_steps)
            self._ep_reward_sum = 0
            self._ep_num_steps = 0
            state = self.env.reset()

        return state[None], np.array(reward)[None], np.array(done)[None], info

    def sample_random_action(self):
        return np.array(self.env.sample_random_action())[None]

    def get_state_info(self):
        info = self.env.get_state_info()
        info.shape = (1,) + info.shape
        return info

    def get_action_info(self):
        return self.env.get_action_info()

    def close(self):
        self.env.close()
