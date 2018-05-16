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
    '''

    def __init__(self, model, env, batch_size=64, num_epochs=10, clip_range=0.2,
                 **kwargs):
        super().__init__(model=model, env=env, **kwargs)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.clip_range = clip_range

    @property
    def loss_fn(self):
        return F.mse_loss
        # return F.smooth_l1_loss

    def clipped_loss(self, batch):
        pred = self.forward(batch.state_t).view(-1)
        pred_clipped = batch.old_pred + (pred - batch.old_pred).clamp(
            -self.clip_range, self.clip_range)

        losses = (pred - batch.vtarget)**2
        losses_clipped = (pred_clipped - batch.vtarget)**2

        loss = 0.5 * torch.max(losses, losses_clipped).mean()

        self.losses.append(loss)

    def add_losses(self, batch):
        self.clipped_loss(batch)

    def train(self, batch):
        batch = batch.apply_to_all(self._to_tensor)

        with torch.no_grad():
            old_preds = self.forward(batch.state_t).view(-1)

        dataset = TensorDataset(batch.state_t, batch.vtarget, old_preds)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.num_epochs):
            for state_t, vtarget, old_pred in data_loader:
                mini_batch = U.Batch(
                    dict(state_t=state_t, vtarget=vtarget, old_pred=old_pred))
                loss = self.optimizer_step(mini_batch)

                if self.logger is not None:
                    self.logger.add_log('Value NN/Loss', loss.item(), precision=3)

        # Ev after update
        if self.logger is not None:
            self.logger.add_log('Value NN/Old Explained Var',
                                U.explained_var(batch.vtarget, old_preds).item())
            preds = self.forward(batch.state_t)
            self.logger.add_log('Value NN/New Explained Var',
                                U.explained_var(batch.vtarget, preds).item())
