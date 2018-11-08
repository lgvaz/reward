from abc import ABC, abstractmethod
import numpy as np
import cv2
import reward.utils as U
from reward.env.wrappers import BaseWrapper


class BaseStateWrapper(BaseWrapper, ABC):
    def __init__(self, env):
        super().__init__(env=env)
        self._shape = None

    @abstractmethod
    def transform(self, state):
        pass

    @property
    def state_space(self):
        space = self.env.state_space
        if self._shape is None:
            self._shape = self.reset().shape

        space.shape = self._shape
        return space

    def reset(self):
        state = self.env.reset()
        return self.transform(state)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = self.transform(state)

        return state, reward, done, info


class RGB2GRAY(BaseStateWrapper):
    def transform(self, state):
        return cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)[..., None]


class Rescale(BaseStateWrapper):
    def __init__(self, env, shape):
        super().__init__(env=env)
        self.shape = shape

    def transform(self, state):
        assert state.ndim == 3 or state.ndim == 2
        state = cv2.resize(state, self.shape, interpolation=cv2.INTER_NEAREST)

        return state if state.ndim == 3 else state[:, :, None]


class HWC2CHW(BaseStateWrapper):
    def transform(self, state):
        assert state.ndim == 3, "frame have {} dims but must have 3".format(state.ndim)
        return np.rollaxis(state, -1)
