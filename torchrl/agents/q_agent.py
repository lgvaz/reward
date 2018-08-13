import numpy as np
import torch
import torchrl.utils as U
from torchrl.agents import BaseAgent


class QAgent(BaseAgent):
    def __init__(
        self,
        batcher,
        *,
        q_model,
        vtarget=U.estimators.value.TDTarget(gamma=0.99),
        **kwargs
    ):
        super().__init__(batcher=batcher, **kwargs)

        self.vtarget = vtarget

        self.register_model("policy", q_model)

    def step(self):
        batch = self.generate_batch()
        batch.state_t = U.to_tensor(batch.state_t)
        batch.state_tp1 = U.to_tensor(batch.state_tp1)

        # self.add_vtarget(batch)

        batch = batch.concat_batch()

        self.train_models(batch)

    # def add_vtarget(self, batch):
    #     state_tp1 = U.join_first_dims(batch.state_tp1, num_dims=2)
    #     q_tp1 = self.models.policy(state_tp1)
    #     max_q_tp1 = q_tp1.max(dim=1)[0]

    #     batch.state_value_tp1 = U.to_np(max_q_tp1)
    #     batch.vtarget = self.vtarget(batch)
