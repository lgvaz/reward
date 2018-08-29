import torch
import torch.nn as nn
import torchrl as tr
import torch.nn.functional as F
from torchrl.models import TargetModel
from torchrl.nn import FlattenLinear


class DDPGCritic(TargetModel):
    @property
    def batch_keys(self):
        return ["state_t", "q_target"]

    def register_losses(self):
        self.register_loss(self.critic_loss)

    # TODO: Can be removed?? Look at others QModel too
    # TODO: Using memory instead of batch will have problems with mini_batches
    # def add_target_value(self, batch):
    #     with torch.no_grad():
    #         self.memory.target_value = self.target_nn(
    #             input=(batch.state_tp1, batch.target_action)
    #         )

    def critic_loss(self, batch):
        pred = self.forward((batch.state_t, batch.action)).squeeze()
        loss = F.mse_loss(input=pred, target=batch.q_target)
        return loss

    def write_logs(self, batch):
        self.eval()

        pred = self.forward((batch.state_t, batch.action))
        target_nn = self.forward_target((batch.state_t, batch.action))
        self.add_histogram_log("pred", pred)
        self.add_histogram_log("target", batch.q_target)
        self.add_histogram_log("target_nn", target_nn)

        self.train()

    @staticmethod
    def output_layer(input_shape, action_shape, action_space):
        if action_space != "continuous":
            raise ValueError(
                "Only works with continuous actions, got {}".format(action_space)
            )
        layer = FlattenLinear(in_features=input_shape, out_features=1)
        layer.weight.data.uniform_(-3e-3, 3e-3)
        return layer
