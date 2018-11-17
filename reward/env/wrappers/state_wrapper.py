import numpy as np
import cv2
import reward.utils as U
from abc import ABC, abstractmethod
from reward.env.wrappers import BaseWrapper
from boltons.cacheutils import cachedproperty


class BaseStateWrapper(BaseWrapper, ABC):
    def __init__(self, env):
        super().__init__(env=env)
        self._shape = None

    @abstractmethod
    def transform(self, s):
        pass

    @cachedproperty
    def s_space(self):
        space = self.env.s_space
        if self._shape is None:
            self._shape = self.reset().shape

        space.shape = self._shape
        return space

    def reset(self):
        s = self.env.reset()
        return self.transform(s)

    def step(self, ac):
        s, r, d, info = self.env.step(ac)
        s = self.transform(s)

        return s, r, d, info


class RGB2GRAY(BaseStateWrapper):
    def transform(self, s):
        return cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)[..., None]


class Rescale(BaseStateWrapper):
    def __init__(self, env, shape):
        super().__init__(env=env)
        self.shape = shape

    def transform(self, s):
        assert s.ndim == 3 or s.ndim == 2
        s = cv2.resize(s, self.shape, interpolation=cv2.INTER_NEAREST)

        return s if s.ndim == 3 else s[:, :, None]


class HWC2CHW(BaseStateWrapper):
    def transform(self, s):
        assert s.ndim == 3, "frame have {} dims but must have 3".format(s.ndim)
        return np.rollaxis(s, -1)
