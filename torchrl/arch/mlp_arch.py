import torch.nn as nn
from torchrl.arch import BaseArch


class MLP(BaseArch):
    def __init__(
        self,
        state_shape,
        action_shape,
        action_space,
        output_layer,
        hidden=[64, 64],
        activation=nn.Tanh,
    ):
        self.hidden = hidden
        self.activation = activation
        super().__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            action_space=action_space,
            output_layer=output_layer,
        )

    def create_body(self):
        layers = []
        layers += [nn.Linear(self.state_shape[0], self.hidden[0]), self.activation()]
        for i in range(1, len(self.hidden)):
            layers += [nn.Linear(self.hidden[i - 1], self.hidden[i]), self.activation()]

        return nn.Sequential(*layers)

    def create_head(self):
        return self.get_output_layer(input_shape=self.hidden[-1])
