from abc import ABC, abstractmethod

import numpy as np
from tqdm.autonotebook import tqdm

import reward.utils as U


class BaseRunner(ABC):
    def __init__(self, env, ep_maxlen=None):
        self.env = env
        self.ep_maxlen = ep_maxlen or float("inf")
        self.rewards = []
        self.num_steps = 0
        self.ep_lens = []
        self.new_ep = 0
        self._is_best = False
        self._last_logged_ep = 0
        self._best_rew = 0

    @property
    @abstractmethod
    def env_name(self):
        pass

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

    @abstractmethod
    def sample_random_action(self):
        pass

    def _wrap_name(self, s):
        return "/".join([self.__class__.__name__, s])

    @property
    def num_episodes(self):
        return len(self.rewards)

    @property
    def is_best(self):
        return self._is_best

    def close(self):
        raise NotImplementedError

    def write_logs(self, logger):
        new_ep = abs(self._last_logged_ep - self.num_episodes)
        if new_ep != 0:
            self.new_ep = new_ep
            self._last_logged_ep = self.num_episodes

        rew = np.mean(self.rewards[-self.new_ep :])
        self._is_best = rew >= self._best_rew
        self._best_rew = max(self._best_rew, rew)

        logger.add_log(self._wrap_name("Reward"), rew)
        logger.add_log(self._wrap_name("Length"), np.mean(self.ep_lens[-self.new_ep :]))
