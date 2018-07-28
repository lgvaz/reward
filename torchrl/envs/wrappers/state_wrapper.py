from abc import ABC, abstractmethod
import numpy as np
import cv2
import torchrl.utils as U
from torchrl.envs.wrappers import BaseWrapper


class BaseStateWrapper(BaseWrapper, ABC):
    def __init__(self, env):
        super().__init__(env=env)
        self._shape = None

    @abstractmethod
    def transform(self, state):
        pass

    def reset(self):
        state = self.env.reset()
        return self.transform(state)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = self.transform(state)

        return state, reward, done, info

    def get_state_info(self):
        info = self.env.get_state_info()

        if self._shape is None:
            self._shape = self.reset().shape

        info.shape = self._shape

        return info


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
