import numpy as np
from abc import ABC, abstractmethod


class BaseRunner(ABC):
    def __init__(self, env):
        self.env = env
        self._rewards = []
        self._steps = 0
        self._last_logged_ep = 0
        self._new_rewards = []

    @property
    @abstractmethod
    def num_envs(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self, action):
        pass

    @abstractmethod
    def get_state_info(self):
        pass

    @abstractmethod
    def get_action_info(self):
        pass

    @property
    def rewards(self):
        return self._rewards

    @property
    def num_steps(self):
        return self._steps

    @property
    def num_episodes(self):
        return len(self._rewards)

    def close(self):
        raise NotImplementedError

    def write_logs(self, logger):
        new_eps = abs(self._last_logged_ep - self.num_episodes)
        if new_eps != 0:
            self._new_rewards = self.rewards[-new_eps:]
            self._last_logged_ep = self.num_episodes

        logger.add_log('Reward/Episode (New)', np.mean(self._new_rewards))
        logger.add_log('Reward/Episode (Last 50)', np.mean(self.rewards[-50:]))
