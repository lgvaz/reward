import torch
import torch.nn as nn
import torchrl as tr
import torch.nn.functional as F
from torchrl.models import TargetModel
from torchrl.nn import FlattenLinear


class DDPGCriticModel(TargetModel):
    # TODO model_before -> model_before_act
    def __init__(
        self,
        model_before,
        model_after,
        batcher,
        *,
        target_up_freq,
        target_up_weight,
        **kwargs
    ):
        model = DDPGModule(model_before=model_before, model_after=model_after)
        super().__init__(
            model=model,
            batcher=batcher,
            target_up_freq=target_up_freq,
            target_up_weight=target_up_weight,
            **kwargs
        )

    @property
    def batch_keys(self):
        return ["state_t", "q_target"]

    @property
    def body(self):
        raise NotImplementedError

    @property
    def head(self):
        raise NotImplementedError

    def register_losses(self):
        self.register_loss(self.critic_loss)

    # TODO: Can be removed?? Look at others QModel too
    # TODO: Using memory instead of batch will have problems with mini_batches
    # def add_target_value(self, batch):
    #     with torch.no_grad():
    #         self.memory.target_value = self.target_net(
    #             input=(batch.state_tp1, batch.target_action)
    #         )

    def critic_loss(self, batch):
        pred = self((batch.state_t, batch.action)).squeeze()
        loss = F.mse_loss(input=pred, target=batch.q_target)
        return loss

    @classmethod
    def from_config(cls, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def output_layer(input_shape, action_info):
        if action_info.space != "continuous":
            raise ValueError(
                "Only works with continuous actions, got {}".format(action_info.space)
            )
        layer = FlattenLinear(in_features=input_shape, out_features=1)
        layer.weight.data.uniform_(-3e-3, 3e-3)
        return layer


class DDPGModule(nn.Module):
    def __init__(self, model_before, model_after):
        super().__init__()
        self.model_before = model_before
        self.model_after = model_after

    def forward(self, input):
        state, action = input
        x = self.model_before(state)
        # Add actions to the activations
        x = torch.cat((x[:, : -action.shape[1]], action), dim=1)
        x = self.model_after(x)

        return x

    def train(self):
        self.model_before.train()
        self.model_after.train()

    def eval(self):
        self.model_before.eval()
        self.model_after.eval()

    def cuda(self):
        self.model_before.cuda()
        self.model_after.cuda()
