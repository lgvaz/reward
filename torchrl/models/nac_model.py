import pdb
import torch
import torch.nn.functional as F
import torchrl as tr
import torchrl.utils as U
from torchrl.models import DQNModel


class NACModel(DQNModel):
    def __init__(self, nn, batcher, *, entropy_weight, v_loss_coef=1., **kwargs):
        super().__init__(nn=nn, batcher=batcher, **kwargs)
        self.entropy_weight = entropy_weight
        self.v_loss_coef = v_loss_coef
        # TODO TODO
        self.gamma = 0.99

    def register_losses(self):
        self.register_loss(self.nac_loss)

    def register_callbacks(self):
        super().register_callbacks()
        self.callbacks.register_on_train_start(self.add_v_target)

    def v_from_q(self, q):
        assert len(q.shape) == 2
        logsumexp_q = (q / self.entropy_weight).logsumexp(dim=-1)
        return self.entropy_weight * logsumexp_q

    def create_dist(self, state):
        q = self.forward(state)
        v = self.v_from_q(q=q)
        probs = ((q - v[..., None]) / self.entropy_weight).exp()

        return tr.distributions.Categorical(probs=probs)

    def add_v_target(self, batch):
        with torch.no_grad():
            # TODO: Assuming batch.q_tp1 is in batch
            batch.v_target = self.v_from_q(q=batch.q_tp1)

    def nac_loss(self, batch):
        # PG loss
        # q_hat = batch.reward + (1 - batch.done) * self.gamma * batch.v_target

        # # TODO Feed forwarding net 2 times for q values
        # selected_q = self.get_selected_q(batch=batch)
        # q = self.forward(batch.state_t)
        # v = self.v_from_q(q=q)
        # with torch.no_grad():
        #     delta = selected_q - q_hat

        # losses_pg = (selected_q - v) * delta
        # loss_pg = losses_pg.mean()

        # # V loss
        # dist = self.create_dist(state=batch.state_t)
        # v_hat = q_hat + self.entropy_weight * dist.entropy()
        # loss_v = ((v - v_hat) ** 2).mean()

        # return loss_pg + self.v_loss_coef * loss_v

        selected_q = self.get_selected_q(batch=batch)
        target_q = self.target_nn(batch.state_tp1)
        target_v = self.entropy_weight * (target_q / self.entropy_weight).logsumexp(-1)
        with torch.no_grad():
            q_hat = batch.reward + (1 - batch.done) * self.gamma * target_v
            adv = selected_q - q_hat

        losses = selected_q * adv
        loss = losses.mean()

        return loss

    def select_action(self, state, step):
        dist = self.create_dist(state)
        return U.to_np(dist.sample())

    def write_logs(self, batch):
        super().write_logs(batch=batch)
        # TODO: feed forwarding network multiple times, here and on create_dist
        q = self.forward(batch.state_t)
        v = self.v_from_q(q=q)

        dist = self.create_dist(state=batch.state_t)

        self.add_histogram_log("Q", q)
        self.add_histogram_log("probs", dist.probs)
        self.add_tf_only_log("V", v)
        self.add_log("Entropy", dist.entropy().mean())
