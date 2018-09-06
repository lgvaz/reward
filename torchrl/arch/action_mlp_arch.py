import pdb
import torch
import torch.nn as nn
from torchrl.arch import MLP


# TODO: Current no support for CONV
# TODO: before action should not be = body, after action should not be = head
class ActionMLP(MLP):
    def __init__(
        self,
        state_shape,
        action_shape,
        action_space,
        output_layer,
        before=[64],
        after=[64],
        activation=nn.ReLU,
        layer_norm=False,
    ):
        self.before = before
        self.after = after
        self.last_body = self.before[-1] if self.before else state_shape[0]
        super().__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            action_space=action_space,
            output_layer=output_layer,
            hidden=[],
            activation=activation,
            layer_norm=layer_norm,
        )

    def create_body(self):
        layers = self.create_block(input_size=self.state_shape[0], units=self.before)
        return nn.Sequential(*layers)

    def create_head(self):
        input_size = self.last_body + self.action_shape[0]
        layers = self.create_block(input_size=input_size, units=self.after)
        layers += [self.get_output_layer(input_shape=self.after[-1])]
        return nn.Sequential(*layers)

    def forward(self, x):
        s, a = x
        x = self.body(s)
        x = torch.cat((x, a), dim=1)
        x = self.head(x)

        return x
