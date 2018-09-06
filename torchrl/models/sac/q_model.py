import pdb
import torch
import torch.nn.functional as F
import torchrl.utils as U
from torchrl.models import BaseValueModel
from torchrl.nn import FlattenLinear


# TODO: Think of another way of doing this
# Maybe eliminate "batch" on loss and pass explicit parameters, would be less confusing
class Q(BaseValueModel):
    @property
    def batch_keys(self):
        return ["state_t", "qtarget"]

    def register_losses(self):
        self.register_loss(self.mse_loss)

    def register_callbacks(self):
        super().register_callbacks()
        self.callbacks.register_on_epoch_start(self.add_old_pred)

    def mse_loss(self, batch):
        pred = self.forward((batch.state_t, batch.action)).view(-1)
        loss = F.mse_loss(pred, batch.qtarget)
        # pdb.set_trace()
        return loss

    def add_old_pred(self, batch):
        with torch.no_grad():
            batch.old_pred = self.forward((batch.state_t, batch.action)).view(-1)

    def write_logs(self, batch):
        super().write_logs(batch=batch)

        self.memory.new_pred = self.forward((batch.state_t, batch.action))
        # self.add_log(
        #     "Old Explained Var", U.explained_var(batch.qtarget, batch.old_pred)
        # )
        # self.add_log(
        #     "New Explained Var", U.explained_var(batch.qtarget, self.memory.new_pred)
        # )
        self.add_log("Target_mean", batch.qtarget.mean())
        self.add_log("Pred_mean", self.memory.new_pred.mean())

    @staticmethod
    def output_layer(input_shape, action_shape, action_space):
        return FlattenLinear(in_features=input_shape, out_features=1)
