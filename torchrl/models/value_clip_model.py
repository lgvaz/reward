import torch

import torchrl.utils as U
from torchrl.models import ValueModel


class ValueClipModel(ValueModel):
    def __init__(self, model, batcher, clip_range=0.2, **kwargs):
        self.clip_range_fn = U.make_callable(clip_range)

        super().__init__(model=model, batcher=batcher, **kwargs)

    @property
    def batch_keys(self):
        return ["state_t", "old_pred", "vtarget"]

    @property
    def clip_range(self):
        return self.clip_range_fn(self.num_steps)

    def register_losses(self):
        self.register_loss(self.clipped_mse_loss)

    def clipped_mse_loss(self, batch):
        pred = self.forward(batch.state_t).view(-1)
        pred_diff = pred - batch.old_pred
        pred_clipped = batch.old_pred + pred_diff.clamp(
            -self.clip_range, self.clip_range
        )

        losses = (pred - batch.vtarget) ** 2
        losses_clipped = (pred_clipped - batch.vtarget) ** 2
        loss = 0.5 * torch.max(losses, losses_clipped).mean()

        return loss

    def write_logs(self, batch):
        super().write_logs(batch)

        pred_diff = self.memory.new_pred - batch.old_pred
        clip_frac = (abs(pred_diff) > self.clip_range).float().mean()
        self.add_log("Clip Range", self.clip_range)
        self.add_log("Clip Fraction", clip_frac)
