import torchrl as tr
import torch.nn as nn
from torchrl.arch import BaseArch


class A3C(BaseArch):
    def create_body(self):
        layers = []
        layers += [nn.Conv2d(self.state_shape[0], 16, 8, 4), nn.ReLU()]
        layers += [nn.Conv2d(16, 32, 4, 2), nn.ReLU()]
        # AvgPool with output size 7 gives the same output as a 84x84 image input
        layers += [nn.AdaptiveAvgPool2d(7), tr.nn.Flatten()]

        return nn.Sequential(*layers)

    def create_head(self):
        layers = []
        layers += [nn.Linear(32 * 7 * 7, 256), nn.ReLU()]
        layers += [self.get_output_layer(input_shape=256)]

        return nn.Sequential(*layers)
