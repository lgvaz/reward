import torch
import torch.nn as nn

from torchrl.nn.body import SequentialNNBody
from torchrl.nn.head import DenseNNHead


class BaseModel(nn.Module):
    def __init__(self, nn_body, nn_head, cuda_default=True):
        super().__init__()

        self.nn_body = nn_body
        self.nn_head = nn_head

        self.cuda_enabled = cuda_default and torch.cuda.is_available()
        if self.cuda_enabled:
            self.nn_body = self.nn_body.cuda()
            self.nn_head = self.nn_head.cuda()

        self.opt = torch.optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, x):
        return self.nn_head(self.nn_body(x))

    @classmethod
    def from_config(cls, config):
        nn_body = config.model.nn_body.obj.from_config(
            config.model.nn_body.arch.as_dict(),
            kwargs=config.model.nn_body.kwargs.as_dict())
        nn_head = config.model.nn_head.obj(
            # TODO: Hardcoded output shape
            input_shape=nn_body.get_output_shape((1, 84, 84)),
            output_shape=4,
            **config.model.nn_head.kwargs.as_dict())

        return cls(nn_body, nn_head, **config.model.kwargs.as_dict())
