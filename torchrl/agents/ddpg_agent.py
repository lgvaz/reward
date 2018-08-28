import pdb
import torch
import torchrl.utils as U
from torchrl.agents import BaseAgent


class DDPGAgent(BaseAgent):
    def __init__(self, actor, critic, batcher, optimizer, action_fn, **kwargs):
        super().__init__(
            batcher=batcher, optimizer=optimizer, action_fn=action_fn, **kwargs
        )
        self.register_model("actor", actor)
        self.register_model("critic", critic)

    def step(self):
        batch = self.generate_batch()
        # TODO: strange to convert to tensor here
        batch = batch.to_tensor()
        batch.state_t = U.to_tensor(batch.state_t)
        batch.state_tp1 = U.to_tensor(batch.state_tp1)

        self.add_q_target(batch)

        batch = batch.concat_batch()
        self.train_models(batch)

    def add_q_target(self, batch):
        # TODO: Modularize
        with torch.no_grad():
            state_tp1 = U.join_first_dims(batch.state_tp1, num_dims=2)
            act_target = self.models.actor.forward_target(state_tp1)
            q_tp1 = self.models.critic.forward_target((state_tp1, act_target))
            q_tp1 = q_tp1.reshape(batch.reward.shape)
            batch.q_target = batch.reward + (1 - batch.done) * 0.99 * q_tp1
