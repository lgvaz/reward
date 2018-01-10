import torch
from torch.autograd import Variable

from torchrl.nn import SequentialExtended


class SequentialNNBody(SequentialExtended):
    def __init__(self, layers_config, normalize_imgs=False):
        super().__init__(layers_config)

        self.normalize_imgs = normalize_imgs

    def forward(self, x):
        if self.normalize_imgs:
            x /= 255.
        output = super().forward(x)

        return output
