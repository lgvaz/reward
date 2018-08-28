import torch
import numpy as np
import torch.nn.functional as F
import torchrl.utils as U
from torchrl.models import BaseValueModel
from torchrl.nn import FlattenLinear


class QModel(BaseValueModel):
    def __init__(
        self,
        model,
        batcher,
        exploration_rate,
        q_target=U.estimators.q.QLearningTarget(gamma=0.99),
        **kwargs
    ):
        super().__init__(model=model, batcher=batcher, **kwargs)
        self.exploration_rate_fn = U.make_callable(exploration_rate)
        self.q_target_fn = q_target
        self.loss_fn = F.mse_loss

    @property
    def batch_keys(self):
        return ["state_t", "action", "q_target"]

    @property
    def exploration_rate(self):
        return self.exploration_rate_fn(self.num_steps)

    def register_callbacks(self):
        super().register_callbacks()
        self.callbacks.register_on_train_start(self.add_q_target)

    def register_losses(self):
        self.register_loss(self.q_loss)

    def add_q_target(self, batch):
        with torch.no_grad():
            batch.q_tp1 = self.forward(batch.state_tp1)
            batch.q_target = self.q_target_fn(batch)

    def get_selected_q(self, batch):
        q_values = self.forward(batch.state_t)
        selected_q = q_values.gather(
            dim=1, index=batch.action.reshape(-1, 1).long()
        ).squeeze()
        return selected_q

    def q_loss(self, batch):
        selected_q = self.get_selected_q(batch)
        loss = self.loss_fn(input=selected_q, target=batch.q_target)
        return loss

    def write_logs(self, batch):
        super().write_logs(batch=batch)
        self.add_tf_only_log(name="Exploration_rate", value=self.exploration_rate)

    @staticmethod
    def output_layer(input_shape, action_info):
        # TODO: Rethink about ActionLinear
        if action_info.space != "discrete":
            raise ValueError(
                "Only works with discrete actions, got {}".format(action_info.space)
            )
        return FlattenLinear(in_features=input_shape, out_features=action_info.shape)

    @staticmethod
    def select_action(model, state, step, training=True):
        if np.random.random() <= model.exploration_rate and training:
            return model.batcher.runner.sample_random_action()
        else:
            # TODO: Calculate pct of actions taken
            with torch.no_grad():
                q_values = model(state)
            return U.to_np(q_values.argmax(dim=1))
