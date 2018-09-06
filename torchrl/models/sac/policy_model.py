import pdb
import torch
import numpy as np
import torch.nn as nn
import torchrl as tr
import torchrl.utils as U
from torchrl.models import BaseModel
from torchrl.nn import FlattenLinear


class Policy(BaseModel):
    def __init__(
        self,
        nn,
        batcher,
        *,
        mean_loss_weight=1e-3,
        std_loss_weight=1e-3,
        pre_activation_weight=0.,
        **kwargs
    ):
        super().__init__(nn=nn, batcher=batcher, **kwargs)
        self.mean_loss_weight = mean_loss_weight
        self.std_loss_weight = std_loss_weight
        self.pre_activation_weight = pre_activation_weight

    @property
    def batch_keys(self):
        return ["dist", "new_action", "q_t_new_act", "new_pre_activation"]

    def register_losses(self):
        # self.register_loss(self.policy_loss)
        self.register_loss(self.policy_kl_loss)
        self.register_loss(self.mean_l2_loss)
        self.register_loss(self.std_l2_loss)
        self.register_loss(self.pre_activation_l2_loss)

    # def policy_loss(self, batch):
    #     log_prob = batch.new_dist.log_prob(batch.new_action).sum(-1)
    #     log_policy_target = batch.q_t_new_act - batch.v_t
    #     losses = log_prob * (log_prob - log_policy_target).detach()
    #     loss = losses.mean()
    #     pdb.set_trace()
    #     return loss

    def policy_kl_loss(self, batch):
        log_prob = batch.new_dist.log_prob(batch.new_action).sum(-1)
        log_target = batch.q_t_new_act

        losses = log_prob - log_target.squeeze()
        loss = losses.mean()
        # pdb.set_trace()
        return loss

    def mean_l2_loss(self, batch):
        losses = self.mean_loss_weight * (batch.new_dist.loc ** 2) * 0.5
        loss = losses.mean()
        return loss

    def std_l2_loss(self, batch):
        losses = self.std_loss_weight * (batch.new_dist.scale.log() ** 2) * 0.5
        loss = losses.mean()
        return loss

    def pre_activation_l2_loss(self, batch):
        losses = self.pre_activation_weight * (batch.new_pre_activation ** 2) * 0.5
        # TODO: Why sum??
        loss = losses.mean()
        return loss

    def create_dist(self, state):
        parameters = self.forward(state)
        mean, log_std = parameters

        return tr.distributions.TanhNormal(loc=mean, scale=log_std.exp())

    def select_action(self, state, step):
        dist = self.create_dist(state=state)
        # TODO: if training don't sample, just take the mean
        if self.training:
            action = U.to_np(dist.sample())
        if not self.training:
            action = U.to_np(dist.sample_det())
        assert not np.isnan(action).any()

        return action

    def write_logs(self, batch):
        super().write_logs(batch=batch)
        dist = self.create_dist(state=batch.state_t)
        replay_log_prob = dist.log_prob(batch.action).sum(-1)
        new_log_prob = dist.log_prob(batch.new_action).sum(-1)

        self.add_histogram_log("dist/mean", dist.loc)
        self.add_histogram_log("dist/std", dist.scale)
        self.add_histogram_log("action/replay", batch.action)
        self.add_histogram_log("action/now", batch.new_action)
        self.add_histogram_log("action/new_log_prob", new_log_prob)
        self.add_histogram_log("action/replay_log_prob", replay_log_prob)

    @staticmethod
    def output_layer(input_shape, action_shape, action_space):
        if action_space != "continuous":
            raise ValueError

        return OutputLayer(input_shape=input_shape, action_shape=action_shape)


class OutputLayer(nn.Module):
    def __init__(self, input_shape, action_shape, w_init=3e-3):
        super().__init__()
        self.w_init = w_init

        out = action_shape[0]
        self.mean = FlattenLinear(in_features=input_shape, out_features=out)
        self.log_std = FlattenLinear(in_features=input_shape, out_features=out)
        # self.log_std = nn.Parameter(torch.zeros(1, out))
        self.init_layer(self.mean)
        # self.init_layer(self.log_std)

    def forward(self, x):
        # TODO: In normal PG is stacked on -1
        # mean = self.mean(x)
        # log_std = self.log_std.expand_as(mean)
        # return torch.stack((mean, log_std), dim=0)
        return torch.stack((self.mean(x), self.log_std(x)), dim=0)

    def init_layer(self, layer):
        layer.weight.data.uniform_(-self.w_init, self.w_init)
        layer.bias.data.uniform_(-self.w_init, self.w_init)
