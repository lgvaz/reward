import torch
import torch.nn.functional as F
import torchrl.utils as U
from copy import deepcopy
from tqdm import tqdm
from torchrl.models import QModel, TargetModel


class DQNModel(QModel, TargetModel):
    def __init__(
        self,
        nn,
        batcher,
        *,
        exploration_rate,
        target_up_freq,
        target_up_weight=1.,
        **kwargs
    ):
        super().__init__(
            nn=nn,
            batcher=batcher,
            exploration_rate=exploration_rate,
            target_up_freq=target_up_freq,
            target_up_weight=target_up_weight,
            **kwargs
        )
        self.loss_fn = F.smooth_l1_loss

    def add_q_target(self, batch):
        with torch.no_grad():
            batch.q_tp1 = self.memory.target_value
            batch.q_target = self.q_target_fn(batch)

    def write_logs(self, batch):
        super().write_logs(batch=batch)
        self.add_histogram_log(name="Q_target", values=batch.q_tp1)
