import torch
from abc import ABC, abstractmethod
from torchrl.nn import ModuleExtended


class BaseModel(ModuleExtended, ABC):
    '''
    Basic TorchRL model. The model should take two PyTorch networks (body and head)
    and chain them together.

    Parameters
    ----------
    nn_body: torch.nn.Module
        The body of the model, should receive the state
        and return a representation used by the head network.
    nn_head: torch.nn.Module
        The head of the model, should receive the outputs of
        the body network and outputs values used for selecting an action.
    cuda_default: bool
        If True and cuda is supported, use it.
    '''

    def __init__(self, nn_body, nn_head, cuda_default=True):
        super().__init__()

        self.nn_body = nn_body
        self.nn_head = nn_head

        self.cuda_enabled = cuda_default and torch.cuda.is_available()
        if self.cuda_enabled:
            self.nn_body = self.nn_body.cuda()
            self.nn_head = self.nn_head.cuda()

        self.opt = self._create_optimizer()
        self.num_updates = 0

    def _create_optimizer(self):
        '''
        Creates an optimizer for the model.

        Returns
        -------
        torch.optim
            A pytorch optimizer.

        Examples
        --------
        It's possible to create an optimizer with the same
        configurations for all the model::

            torch.optim.Adam(self.parameters(), lr=1e-2)

        Or use a different configuration for different parts of the model::

            torch.optim.Adam(
                [
                    dict(params=self.nn_body.parameters(), lr=1e-3),
                    dict(params=self.nn_head.parameters(), epsilon=1e-7)
                ],
                lr=1e-2)

        For more information see
        `here <http://pytorch.org/docs/0.3.0/optim.html#per-parameter-options>`_.
        '''
        return torch.optim.Adam(self.parameters(), lr=1e-2)
        # return torch.optim.Adam(
        #     [
        #         dict(params=self.nn_body.parameters()),
        #         dict(params=self.nn_head.parameters())
        #     ],
        #     lr=1e-2)

    @abstractmethod
    def select_action(self, state):
        '''
        This method should be overwritten by a subclass.

        It should receive the state and select an action based on it.

        Returns
        -------
        action: int or numpy.ndarray
        '''
        pass

    @abstractmethod
    def train(self, batch=None):
        '''
        This method should be inherited by a subclass.

        Should use the batch to compute and apply gradients to the network.

        Parameters
        ----------
        batch: dict
            The batch should contain all the information necessary
            to compute the gradients.
        '''
        self.num_updates += 1

    def forward(self, x):
        '''
        Feeds the output of the body network directly into the head.

        Parameters
        ----------
        x: numpy.ndarray
            The environment state.
        '''
        return self.nn_head(self.nn_body(x))

    @classmethod
    def from_config(cls, config, state_shape, action_shape):
        '''
        Creates a model from a configuration file.

        Returns
        -------
        torchrl.models
            A TorchRL model.
        '''
        nn_body = config.model.nn_body.obj.from_config(
            config.model.nn_body.arch.as_dict(),
            kwargs=config.model.nn_body.kwargs.as_dict())
        nn_head = config.model.nn_head.obj(
            input_shape=nn_body.get_output_shape(state_shape),
            output_shape=action_shape,
            **config.model.nn_head.kwargs.as_dict())

        return cls(nn_body, nn_head, **config.model.kwargs.as_dict())
