import torch
import torch.nn.functional as F
import torchrl.utils as U
from copy import deepcopy
from torchrl.models import QModel


class DQNModel(QModel):
    def __init__(
        self,
        model,
        batcher,
        *,
        exploration_rate,
        target_up_freq,
        target_up_weight=1.,
        **kwargs
    ):
        super().__init__(
            model=model, batcher=batcher, exploration_rate=exploration_rate, **kwargs
        )
        self.target_up_freq = target_up_freq
        self.target_up_weight = target_up_weight
        self.loss_fn = F.smooth_l1_loss

        self.target_net = deepcopy(self.model)
        self.target_net.eval()

    def add_q_target(self, batch):
        with torch.no_grad():
            batch.q_tp1 = self.target_net(batch.state_tp1)
            batch.q_target = self.q_target_fn(batch)

    def update_target_net(self, weight):
        if weight == 1.:
            self.target_net.load_state_dict(self.model.state_dict())
        else:
            for fp, tp in zip(self.model.parameters(), self.target_net.parameters()):
                v = weight * fp + (1 - weight) * tp
                tp.data.copy_(v)
