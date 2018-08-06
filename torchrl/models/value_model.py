import torch
import torchrl.utils as U
from torchrl.models import BaseValueModel
from torchrl.nn import FlattenLinear


class ValueModel(BaseValueModel):
    """
    A standard regression model, can be used to estimate the value of states or Q values.

    Parameters
    ----------
    clip_range: float
        Similar to PPOClip, limits the change between the new and old value function.
    """

    @property
    def batch_keys(self):
        return ["state_t", "vtarget"]

    def register_losses(self):
        self.register_loss(self.mse_loss)

    def register_callbacks(self):
        super().register_callbacks()
        self.callbacks.register_on_epoch_start(self.add_old_pred)

    def mse_loss(self, batch):
        pred = self.forward(batch.state_t).view(-1)
        loss = F.mse_loss(pred, batch.vtarget)
        return loss

    def add_old_pred(self, batch):
        with torch.no_grad():
            batch.old_pred = self.forward(batch.state_t).view(-1)

    def write_logs(self, batch):
        super().write_logs(batch)

        self.memory.new_pred = self.forward(batch.state_t)
        self.add_log(
            "Old Explained Var", U.explained_var(batch.vtarget, batch.old_pred)
        )
        self.add_log(
            "New Explained Var", U.explained_var(batch.vtarget, self.memory.new_pred)
        )
        self.add_log("Target_mean", batch.vtarget.mean())
        self.add_log("Pred_mean", self.memory.new_pred.mean())

    @staticmethod
    def output_layer(input_shape, action_info):
        return FlattenLinear(in_features=input_shape, out_features=1)
