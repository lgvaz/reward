import torch
from copy import deepcopy
from torchrl.models import BaseModel


class TargetModel(BaseModel):
    def __init__(self, nn, batcher, *, target_up_freq, target_up_weight, **kwargs):
        super().__init__(nn=nn, batcher=batcher, **kwargs)
        self.target_up_freq = target_up_freq
        self.target_up_weight = target_up_weight
        self.last_target_up = 0

        self.target_nn = deepcopy(nn)
        self.target_nn.eval()
        self.freeze_target()

    def forward_target(self, x):
        return self.target_nn(x)

    def freeze_target(self):
        for param in self.target_nn.parameters():
            param.requires_grad = False

    def unfreeze_target(self):
        for param in self.target_nn.parameters():
            param.requires_grad = True

    def register_callbacks(self):
        # TODO TODO TODO TODO TODO
        # self.callbacks.register_on_train_start(self.add_target_value)
        self.callbacks.register_on_train_start(self.update_target_nn_callback)
        super().register_callbacks()

    def add_target_value(self, batch):
        with torch.no_grad():
            self.memory.target_value = self.target_nn(batch.state_tp1)

    def update_target_nn_callback(self, batch):
        if abs(self.num_steps - self.last_target_up) >= self.target_up_freq:
            self.last_target_up = self.num_steps
            self.update_target_nn(weight=self.target_up_weight)

    def update_target_nn(self, weight):
        if weight == 1.:
            self.target_nn.load_state_dict(self.nn.state_dict())
        else:
            for fp, tp in zip(self.nn.parameters(), self.target_nn.parameters()):
                v = weight * fp + (1 - weight) * tp
                tp.data.copy_(v)

    def write_logs(self, batch):
        super().write_logs(batch=batch)
        values = self.forward_target(batch.state_t)
        self.add_histogram_log("target_nn_values", values)
