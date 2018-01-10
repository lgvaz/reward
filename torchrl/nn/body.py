import torch
from torch.autograd import Variable

from torchrl.nn import SequentialExtended


class SequentialNNBody(SequentialExtended):
    def __init__(self, layers_config, normalize_imgs=False):
        super().__init__(layers_config)

        self.normalize_imgs = normalize_imgs

    def forward(self, x):
        # x = Variable(self._maybe_cuda(torch.from_numpy(x).float()))
        if self.normalize_imgs:
            x /= 255.
        # output = self.layers(x)
        output = super().forward(x)

        return output
