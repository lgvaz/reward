import torch
from torchrl.models import SurrogatePGModel


class PPOModel(SurrogatePGModel):
    def __init__(self,
                 policy_nn_config,
                 value_nn_config=None,
                 share_body=False,
                 ppo_clip_range=0.2,
                 kl_penalty_coef=1,
                 kl_target=0.01,
                 **kwargs):
        self.ppo_clip_range = ppo_clip_range
        self.kl_penalty_coef = kl_penalty_coef
        self.kl_target = kl_target

        super().__init__(
            policy_nn_config=policy_nn_config,
            value_nn_config=value_nn_config,
            share_body=share_body,
            **kwargs)

    def add_ppo_clip(self, batch, new_dists):
        prob_ratio = self.calculate_prob_ratio(batch, new_dists)
        surrogate = prob_ratio * batch['advantages']

        clipped_prob_ratio = prob_ratio.clamp(
            min=1 - self.ppo_clip_range, max=1 + self.ppo_clip_range)
        clipped_surrogate = clipped_prob_ratio * batch['advantages']

        losses = torch.min(surrogate, clipped_surrogate)
        loss = -losses.sum()

        self.losses.append(loss)

    def add_ppo_adaptive_kl(self, batch, new_dists):
        prob_ratio = self.calculate_prob_ratio(batch, new_dists)
        surrogate = prob_ratio * batch['advantages']

        kl_div = self.kl_divergence(new_dists)
        kl_loss = self.kl_penalty_coef * kl_div
        hinge_loss = 1000 * torch.clamp(kl_div - 2 * self.kl_target, min=0)**2

        losses = surrogate - kl_loss - hinge_loss
        loss = -losses.sum()

        self.losses.append(loss)

    def add_losses(self, batch):
        new_dists = self.create_new_dists(batch['state_ts'])

        self.add_ppo_clip(batch, new_dists)
        # self.add_ppo_adaptive_kl(batch, new_dists)
        self.add_value_nn_loss(batch)

    def train(self, batch, num_epochs=10, logger=None):
        super().train(batch=batch, num_epochs=num_epochs, logger=logger)
