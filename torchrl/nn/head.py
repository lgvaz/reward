from collections import OrderedDict

import torch.nn as nn

from torchrl.nn import SequentialExtended


class DenseNNHead(SequentialExtended):
    def __init__(self, input_shape, output_shape, units=[], activation=nn.ReLU):
        in_units = input_shape.prod()
        units = units.copy()
        units.insert(0, in_units)
        units.append(output_shape)

        layers_config = OrderedDict()
        for i_layer in range(len(units) - 1):
            layers_config['linear{}'.format(i_layer + 1)] = nn.Linear(
                in_features=units[i_layer], out_features=units[i_layer + 1])
            # Apply the activation function to all but the last layer
            if i_layer < len(units) - 2:
                layers_config['activation{}'.format(i_layer + 1)] = activation()

        super().__init__(layers_config)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        return super().forward(x)
