import torch.nn.functional as F
import torchrl.utils as U
from torchrl.models import BaseModel


class ValueModel(BaseModel):
    @property
    def loss_fn(self):
        return F.mse_loss

    def add_losses(self, batch):
        preds = self.forward(batch.state_ts).view(-1)
        self.losses.append(self.loss_fn(preds, self._to_tensor(batch.vtargets)))

    def train(self, batch):
        self.optimizer_step(batch)
