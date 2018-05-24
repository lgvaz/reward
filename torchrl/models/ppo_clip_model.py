import torch
import torchrl.utils as U
from torchrl.models import SurrogatePGModel


class PPOClipModel(SurrogatePGModel):
    '''
    Proximal Policy Optimization as described in https://arxiv.org/pdf/1707.06347.pdf.


    Parameters
    ----------
    ppo_clip_range: float
        Clipping value for the probability ratio (Default is 0.2).
    num_epochs: int
        How many times to train over the entire dataset (Default is 10).
    '''

    def __init__(self, model, env, ppo_clip_range=0.2, num_epochs=10, **kwargs):
        super().__init__(model=model, env=env, num_epochs=num_epochs, **kwargs)
        self.ppo_clip_range = U.make_callable(ppo_clip_range)

    def add_losses(self, batch):
        self.ppo_clip_loss(batch)
        self.entropy_loss(batch)

    def ppo_clip_loss(self, batch):
        '''
        Calculate the PPO Clip loss as described in the paper.

        Parameters
        ----------
            batch: Batch
        '''
        self.memory.prob_ratio = self.calculate_prob_ratio(batch.new_log_prob,
                                                           batch.log_prob)
        clipped_prob_ratio = self.memory.prob_ratio.clamp(
            1 - self.ppo_clip_range(self.step), 1 + self.ppo_clip_range(self.step))

        surrogate = self.memory.prob_ratio * batch.advantage
        clipped_surrogate = clipped_prob_ratio * batch.advantage

        losses = torch.min(surrogate, clipped_surrogate)
        loss = -losses.mean()

        self.losses.append(loss)

    def write_logs(self, batch):
        super().write_logs(batch)

        clip_frac = ((1 - self.memory.prob_ratio).abs() > self.ppo_clip_range(
            self.step)).float().mean()

        self.logger.add_log(self.name + '/PPO Clip Range', self.ppo_clip_range(self.step))
        self.logger.add_log(self.name + '/PPO Clip Fraction', clip_frac)
