import torch
import numpy as np
import torch.nn.functional as F
import torchrl.utils as U
from torchrl.models import BaseValueModel
from torchrl.nn import FlattenLinear


class QModel(BaseValueModel):
    def __init__(self, model, batcher, exploration_rate, **kwargs):
        super().__init__(model=model, batcher=batcher, **kwargs)
        self.exploration_rate_fn = U.make_callable(exploration_rate)

    @property
    def batch_keys(self):
        return ["state_t", "action", "vtarget"]

    @property
    def exploration_rate(self):
        return self.exploration_rate_fn(self.num_steps)

    def register_losses(self):
        self.register_loss(self.huber_loss)

    def huber_loss(self, batch):
        q_values = self.forward(batch.state_t)
        selected_q = q_values.gather(dim=1, index=batch.action.reshape(-1, 1)).squeeze()
        loss = F.smooth_l1_loss(selected_q, batch.vtarget)
        return loss

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
            with torch.no_grad():
                q_values = model(state)
            return U.to_np(q_values.argmax(dim=1))
