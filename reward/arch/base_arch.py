from abc import ABC, abstractmethod
import torch.nn as nn


class BaseArch(nn.Module, ABC):
    def __init__(self, state_shape, action_shape, action_space, output_layer):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_space = action_space
        self.output_layer = output_layer

        self.body = self.create_body()
        self.head = self.create_head()

    @abstractmethod
    def create_body(self):
        pass

    @abstractmethod
    def create_head(self):
        pass

    def get_output_layer(self, input_shape):
        return self.output_layer(
            input_shape=input_shape,
            action_shape=self.action_shape,
            action_space=self.action_space,
        )

    def forward(self, x):
        return self.head(self.body(x))

    @classmethod
    def from_env(cls, env, output_layer, **kwargs):
        return cls(
            state_shape=env.state_info.shape,
            action_shape=env.action_info.shape,
            action_space=env.action_info.space,
            output_layer=output_layer,
            **kwargs
        )

    @classmethod
    def from_batcher(cls, batcher, output_layer, **kwargs):
        return cls(
            state_shape=batcher.get_state_info().shape,
            action_shape=batcher.get_action_info().shape,
            action_space=batcher.get_action_info().space,
            output_layer=output_layer,
        )
