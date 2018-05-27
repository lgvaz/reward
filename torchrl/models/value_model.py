import torch
import torch.nn.functional as F
import torchrl.utils as U
from torchrl.models import BaseModel

from torch.utils.data import TensorDataset, DataLoader


class ValueModel(BaseModel):
    '''
    A standard regression model, can be used to estimate the value of states or Q values.

    Parameters
    ----------
    batch_size: int
        The mini-batch size (Default is 64).
    num_epochs: int
        How many times to train over the entire dataset (Default is 10).
    clip_range: float
        Similar to PPOClip, limits the change between the new and old value function.
    '''

    def __init__(self,
                 model,
                 env,
                 num_mini_batches=4,
                 num_epochs=10,
                 clip_range=None,
                 **kwargs):
        super().__init__(model=model, env=env, **kwargs)
        self.num_mini_batches = num_mini_batches
        self.num_epochs = num_epochs
        self.clip_range = U.make_callable(clip_range)

        assert clip_range is None or clip_range > 0, 'clip_range must be None or > 0'

    def mse_loss(self, batch):
        pred = self.forward(batch.state_t).view(-1)
        loss = F.mse_loss(pred, batch.vtarget)

        self.losses.append(loss)

    def clipped_mse_loss(self, batch):
        pred = self.forward(batch.state_t).view(-1)
        pred_diff = pred - batch.old_pred
        pred_clipped = batch.old_pred + pred_diff.clamp(-self.clip_range(self.step),
                                                        self.clip_range(self.step))

        losses = (pred - batch.vtarget)**2
        losses_clipped = (pred_clipped - batch.vtarget)**2

        loss = 0.5 * torch.max(losses, losses_clipped).mean()

        self.losses.append(loss)

    def add_losses(self, batch):
        if self.clip_range(self.step) is None:
            self.mse_loss(batch)
        else:
            self.clipped_mse_loss(batch)

    def train_step(self, batch):
        with torch.no_grad():
            batch.old_pred = self.forward(batch.state_t).view(-1)

        # self.memory.batch_keys.extend(['state_t', 'old_pred', 'vtarget'])
        self.register_batch_keys('state_t', 'old_pred', 'vtarget')
        self.learn_from_batch(batch)

    def write_logs(self, batch):
        super().write_logs(batch)

        self.logger.add_log(self.name + '/Old Explained Var',
                            U.explained_var(batch.vtarget, batch.old_pred))
        # U.explained_var(batch.vtarget, self.memory.old_preds))
        pred = self.forward(batch.state_t)
        self.logger.add_log(self.name + '/New Explained Var',
                            U.explained_var(batch.vtarget, pred))

        pred_diff = pred - batch.old_pred
        # pred_diff = pred - self.memory.old_preds
        clip_frac = (abs(pred_diff) > self.clip_range(self.step)).float().mean()
        self.logger.add_log(self.name + '/Clip Range', self.clip_range(self.step))
        self.logger.add_log(self.name + '/Clip Fraction', clip_frac)
