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

    def train(self, batch, batch_size=64, num_epochs=10):
        batch = batch.apply_to_all(self._to_tensor)

        dataset = TensorDataset(batch.state_ts, batch.vtargets)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(num_epochs):
            loss_sum = 0
            for states, vtargets in data_loader:
                loss = self.optimizer_step(states=states, vtargets=vtargets)
                loss_sum += loss
            print('Value Loss: {}'.format(loss))

        vtargets_mean = batch.vtargets.mean()
        preds_mean = self.forward(batch.state_ts).mean()
        print('vtargets mean: {} | NN mean: {}'.format(vtargets_mean, preds_mean))
