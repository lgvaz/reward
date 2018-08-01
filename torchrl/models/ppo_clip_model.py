import torch
import torchrl.utils as U
from torchrl.models import SurrogatePGModel


class PPOClipModel(SurrogatePGModel):
    """
    Proximal Policy Optimization as described in https://arxiv.org/pdf/1707.06347.pdf.


    Parameters
    ----------
    ppo_clip_range: float
        Clipping value for the probability ratio (Default is 0.2).
    num_epochs: int
        How many times to train over the entire dataset (Default is 10).
    """

    def __init__(self, model, batcher, ppo_clip_range=0.2, **kwargs):
        super().__init__(model=model, batcher=batcher, **kwargs)
        self.ppo_clip_range_fn = U.make_callable(ppo_clip_range)

    @property
    def ppo_clip_range(self):
        return self.ppo_clip_range_fn(self.num_steps)

    def register_losses(self):
        self.register_loss(self.ppo_clip_loss)
        self.register_loss(self.entropy_loss)

    def ppo_clip_loss(self, batch):
        """
        Calculate the PPO Clip loss as described in the paper.

        Parameters
        ----------
            batch: Batch
        """
        clipped_prob_ratio = self.memory.prob_ratio.clamp(
            min=1 - self.ppo_clip_range, max=1 + self.ppo_clip_range
        )

        surrogate = self.memory.prob_ratio * batch.advantage
        clipped_surrogate = clipped_prob_ratio * batch.advantage

        losses = torch.min(surrogate, clipped_surrogate)
        assert len(losses.shape) == 1
        loss = -losses.mean()

        return loss

    def write_logs(self, batch):
        super().write_logs(batch)

        clip_frac = (
            ((1 - self.memory.prob_ratio).abs() > self.ppo_clip_range).float().mean()
        )

        self.add_log("PPO Clip Range", self.ppo_clip_range)
        self.add_log("PPO Clip Fraction", clip_frac)
