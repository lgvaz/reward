import torch
import torch.nn as nn

from torchrl.nn.body import SequentialNNBody
from torchrl.nn.head import DenseNNHead

from collections import OrderedDict
from torchrl.nn import SequentialExtended


class BaseModel(nn.Module):
    def __init__(self, nn_body, nn_head, cuda_default=True):
        super().__init__()

        self.nn_body = nn_body
        self.nn_head = nn_head

        self.cuda_enabled = cuda_default and torch.cuda.is_available()
        if self.cuda_enabled:
            self.nn_body = self.nn_body.cuda()
            self.nn_head = self.nn_head.cuda()

        self.opt = self._create_optimizer()

    def forward(self, x):
        return self.nn_head(self.nn_body(x))
        # return self.nn_body(x)

    def _create_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)
        # return torch.optim.SGD(
        #     [
        #         dict(params=self.nn_body.parameters()),
        #         dict(params=self.nn_head.parameters())
        #     ],
        #     lr=1e-2)

    @classmethod
    def from_config(cls, config, state_shape, action_shape):
        nn_body = config.model.nn_body.obj.from_config(
            config.model.nn_body.arch.as_dict(),
            kwargs=config.model.nn_body.kwargs.as_dict())
        nn_head = config.model.nn_head.obj(
            input_shape=nn_body.get_output_shape(state_shape),
            # input_shape=torch.Tensor([64]).int(),
            output_shape=action_shape,
            **config.model.nn_head.kwargs.as_dict())
        # nn_body = SequentialExtended(
        #     OrderedDict(linear1=nn.Linear(4, 64), relu1=nn.ReLU()))
        # nn_head = SequentialExtended(OrderedDict(linear1=nn.Linear(64, 2)))

        return cls(nn_body, nn_head, **config.model.kwargs.as_dict())
