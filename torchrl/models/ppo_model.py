import torch
from torchrl.models import SurrogatePGModel


class PPOModel(SurrogatePGModel):
    def add_ppo_clip(self, batch):
        prob_ratio = self.calculate_prob_ratio(batch)
        surrogate = prob_ratio * batch['advantages']

        clipped_prob_ratio = prob_ratio.clamp(min=1 - 0.2, max=1 + 0.2)
        clipped_surrogate = clipped_prob_ratio * batch['advantages']

        losses = torch.min(surrogate, clipped_surrogate)
        loss = -losses.sum()

        self.losses.append(loss)

    def add_losses(self, batch):
        self.add_ppo_clip(batch)
        self.add_value_nn_loss(batch)
