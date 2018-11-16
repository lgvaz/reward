from abc import ABC, abstractmethod
from reward.env.wrappers import BaseWrapper


class BaseRewardWrapper(BaseWrapper, ABC):
    @abstractmethod
    def transform(self, r):
        pass

    def step(self, action):
        state, r, done, info = self.env.step(action)
        r = self.transform(r)

        return state, r, done, info
