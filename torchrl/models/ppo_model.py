import torch
from torchrl.models import SurrogatePGModel


class PPOModel(SurrogatePGModel):
    def __init__(self, model, env, ppo_clip_range=0.2, num_epochs=10, **kwargs):
        super().__init__(model=model, env=env, num_epochs=num_epochs, **kwargs)
        self.ppo_clip_range = ppo_clip_range

    def add_losses(self, batch):
        self.ppo_clip_loss(batch)

    def ppo_clip_loss(self, batch):
        prob_ratio = self.calculate_prob_ratio(batch.new_log_prob, batch.log_prob)
        clipped_prob_ratio = prob_ratio.clamp(1 - self.ppo_clip_range,
                                              1 + self.ppo_clip_range)

        surrogate = prob_ratio * batch.advantage
        clipped_surrogate = clipped_prob_ratio * batch.advantage

        losses = torch.min(surrogate, clipped_surrogate)
        loss = -losses.mean()

        self.losses.append(loss)

        if self.logger is not None:
            clip_frac = ((1 - prob_ratio).abs() > self.ppo_clip_range).float().mean()
            self.logger.add_log('Policy/PPO Clip Fraction', clip_frac)
