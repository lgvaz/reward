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
        How many times to train over the entire dataset.
    '''

    def __init__(self, model, env, batch_size=64, num_epochs=10, **kwargs):
        super().__init__(model=model, env=env, **kwargs)
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    @property
    def loss_fn(self):
        # return F.mse_loss
        return F.smooth_l1_loss

    def add_losses(self, states, vtargets):
        preds = self.forward(states).view(-1)
        loss = self.loss_fn(preds, vtargets)
        self.losses.append(loss)

    def train(self, batch):
        batch = batch.apply_to_all(self._to_tensor)

        dataset = TensorDataset(batch.state_t, batch.vtarget)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.num_epochs):
            for states, vtargets in data_loader:
                loss = self.optimizer_step(states=states, vtargets=vtargets)
                if self.logger is not None:
                    self.logger.add_log('Value NN/Loss', loss.item(), precision=3)

        if self.logger is not None:
            preds = self.forward(batch.state_t)
            self.logger.add_log('Value NN/Explained Var',
                                U.explained_var(batch.vtarget, preds).item())
