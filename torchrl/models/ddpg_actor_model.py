import pdb
import numpy as np
import torchrl.utils as U
import torch.nn as nn
from tqdm import tqdm
from torchrl.models import TargetModel
from torchrl.distributions import Ornstein
from torchrl.nn import FlattenLinear


class DDPGActor(TargetModel):
    def __init__(
        self,
        nn,
        batcher,
        critic,
        *,
        target_up_freq,
        target_up_weight,
        action_noise=None,
        **kwargs
    ):
        super().__init__(
            nn=nn,
            batcher=batcher,
            target_up_freq=target_up_freq,
            target_up_weight=target_up_weight,
            **kwargs
        )
        self.critic = critic

        if action_noise is None:
            mu = np.zeros(self.batcher.get_action_info().shape)
            sigma = 0.2 * np.ones(self.batcher.get_action_info().shape)
            self.action_noise = Ornstein(mu=mu, sigma=sigma)

    @property
    def batch_keys(self):
        return ["state_t"]

    def register_losses(self):
        self.register_loss(self.ddpg_loss)

    def ddpg_loss(self, batch):
        action = self.forward(batch.state_t)
        losses = self.critic((batch.state_t, action))
        loss = -losses.mean()

        return loss

    def select_action(self, state, step):
        action = self.nn(state)
        action = U.to_np(action)

        # Explore
        # TODO: Add multiplier to noise (exploration rate)
        if self.training:
            action += self.action_noise.sample()

        return action

    @staticmethod
    def output_layer(input_shape, action_shape, action_space):
        # TODO: Rethink about ActionLinear
        if action_space != "continuous":
            raise ValueError(
                "Only works with continuous actions, got {}".format(action_space)
            )
        tqdm.write("WARNING: Bounding action range between -1 and 1")
        layer = FlattenLinear(in_features=input_shape, out_features=action_shape[0])
        layer.weight.data.uniform_(-3e-3, 3e-3)

        return nn.Sequential(layer, nn.Tanh())
