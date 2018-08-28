import torch
from copy import deepcopy
from torchrl.models import BaseModel


class TargetModel(BaseModel):
    def __init__(self, model, batcher, *, target_up_freq, target_up_weight, **kwargs):
        super().__init__(model=model, batcher=batcher, **kwargs)
        self.target_up_freq = target_up_freq
        self.target_up_weight = target_up_weight
        self.last_target_up = 0

        self.target_net = deepcopy(model)
        self.target_net.eval()

    def forward_target(self, x):
        return self.target_net(x)

    def register_callbacks(self):
        # TODO TODO TODO TODO TODO
        # self.callbacks.register_on_train_start(self.add_target_value)
        self.callbacks.register_on_train_start(self.update_target_net_callback)
        super().register_callbacks()

    def add_target_value(self, batch):
        with torch.no_grad():
            self.memory.target_value = self.target_net(batch.state_tp1)

    def update_target_net_callback(self, batch):
        if abs(self.num_steps - self.last_target_up) >= self.target_up_freq:
            self.last_target_up = self.num_steps
            self.update_target_net(weight=self.target_up_weight)

    def update_target_net(self, weight):
        if weight == 1.:
            self.target_net.load_state_dict(self.model.state_dict())
        else:
            for fp, tp in zip(self.model.parameters(), self.target_net.parameters()):
                v = weight * fp + (1 - weight) * tp
                tp.data.copy_(v)
