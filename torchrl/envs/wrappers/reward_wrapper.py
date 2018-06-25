from abc import ABC, abstractmethod
from torchrl.envs.wrappers import BaseWrapper


class BaseRewardWrapper(BaseWrapper, ABC):
    @abstractmethod
    def transform(self, reward):
        pass

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward = self.transform(reward)

        return state, reward, done, info
