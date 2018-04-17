import torch.nn.functional as F
import torchrl.utils as U
from torchrl.models import BaseModel

from torch.utils.data import TensorDataset, DataLoader


class ValueModel(BaseModel):
    @property
    def loss_fn(self):
        # return F.mse_loss
        return F.smooth_l1_loss

    def add_losses(self, states, vtargets):
        preds = self.forward(states).view(-1)
        loss = self.loss_fn(preds, vtargets)
        self.losses.append(loss)

    def train(self, batch, batch_size=128, num_epochs=7):
        batch = batch.apply_to_all(self._to_tensor)

        dataset = TensorDataset(batch.state_t, batch.vtarget)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(num_epochs):
            for states, vtargets in data_loader:
                loss = self.optimizer_step(states=states, vtargets=vtargets)
                if self.logger is not None:
                    self.logger.add_log('Value NN Loss', loss.item(), precision=3)

        if self.logger is not None:
            preds = self.forward(batch.state_t)
            self.logger.add_log('Value NN EV',
                                U.explained_var(batch.vtarget, preds).item())
