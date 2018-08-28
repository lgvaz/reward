import pdb
import torch
import torch.nn as nn
from torchrl.arch import BaseArch


# TODO: Current no support for CONV
# TODO: before action should not be = body, after action should not be = head
class DDPGCritic(BaseArch):
    def __init__(
        self,
        state_shape,
        action_shape,
        action_space,
        output_layer,
        before=[64],
        after=[64],
        activation=nn.ReLU,
    ):
        self.before = before
        self.after = after
        self.activation = activation
        super().__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            action_space=action_space,
            output_layer=output_layer,
        )

    def create_body(self):
        layers = []
        layers += [nn.Linear(self.state_shape[0], self.before[0]), self.activation()]
        for i in range(1, len(self.before)):
            layers += [nn.Linear(self.before[i - 1], self.before[i]), self.activation()]

        return nn.Sequential(*layers)

    def create_head(self):
        layers = []
        layers += [nn.Linear(self.before[-1] + self.action_shape[0], self.after[-1])]
        for i in range(1, len(self.after)):
            layers += [nn.Linear(self.after[i - 1], self.after[i]), self.activation()]
        layers += [self.get_output_layer(input_shape=self.after[-1])]

        return nn.Sequential(*layers)

    def forward(self, x):
        s, a = x
        x = self.body(s)
        x = torch.cat((x, a), dim=1)
        x = self.head(x)

        return x
