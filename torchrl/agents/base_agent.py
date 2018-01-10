from abc import ABC

from torchrl.utils import get_obj
from torchrl.utils.config import Config

import torch
from torch.autograd import Variable


class BaseAgent(ABC):
    _model = None

    def __init__(self, env, model=None):
        self.env = env
        self.model = model or self._model

    def select_action(self, state):
        # return self.model.select_action(state[None])
        return self.model.select_action(
            Variable(torch.from_numpy(state[None])).float().cuda())

    def run_one_episode(self):
        return self.env.run_one_episode(select_action_fn=self.select_action)

    @classmethod
    def from_config(cls, config):
        env = get_obj(config.env.obj)
        state_shape = env.state_info['shape']
        action_shape = env.action_info['shape']
        model = cls._model.from_config(config, state_shape, action_shape)

        return cls(env, model)

    @classmethod
    def from_file(cls, file_path):
        config = Config.load(file_path)

        return cls.from_config(config)
