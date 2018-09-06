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
        layer_norm=False,
    ):
        self.hidden = hidden
        self.activation = activation
        self.layer_norm = layer_norm
        super().__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            action_space=action_space,
            output_layer=output_layer,
        )

    def create_body(self):
        layers = self.create_block(input_size=self.state_shape[0], units=self.hidden)
        return nn.Sequential(*layers)

    def create_head(self):
        return self.get_output_layer(input_shape=self.hidden[-1])

    def create_block(self, input_size, units):
        layers = []
        if units:
            units.insert(0, input_size)

            for n_in, n_out in zip(units[:-1], units[1:]):
                layers += [nn.Linear(n_in, n_out)]
                if self.layer_norm:
                    layers += [nn.LayerNorm(n_out)]
                layers += [self.activation()]

        return layers
