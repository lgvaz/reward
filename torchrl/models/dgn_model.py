import pdb
import torch
import numpy as np
import torch.nn.functional as F
import torchrl as tr
import torchrl.utils as U
from torchrl.models import DQNModel


# TODO: Many differences from DQN to inherit, exploration_rate, etc
# TODO: Maybe multiple inheratance from GModel and DQN?
class DGNModel(DQNModel):
    def __init__(self, model, batcher, *, prior_weight, prior_policy=None, **kwargs):
        super().__init__(model, batcher, **kwargs)
        self.prior_weight_fn = U.make_callable(prior_weight)
        self.prior_policy = prior_policy or self.default_prior
        # TODO TODO TODO, use estimator for G-learning? Change design?
        self.gamma = 0.99

    @property
    def prior_weight(self):
        return self.prior_weight_fn(self.num_steps)

    def register_callbacks(self):
        super().register_callbacks()

    # TODO TODO: Carefull here for parallel envs, batch would be concatenated
    def add_q_target(self, batch):
        with torch.no_grad():
            g_tp1 = self.target_net(batch.state_tp1)
            prior_dist = self.prior_policy(state=batch.state_tp1)
            prior_probs = prior_dist.probs
            assert g_tp1.shape == prior_probs.shape

            expected_g_tp1 = (
                prior_probs.log() + (g_tp1 / self.prior_weight)
            ).logsumexp(dim=-1)
            # g_exp = (g_tp1 / self.prior_weight).exp()
            # expected_g_tp1 = (prior_probs * g_exp).sum(-1).log()
            assert len(expected_g_tp1.shape) == 1

            g_target = (
                batch.reward
                + (1 - batch.done) * self.gamma * self.prior_weight * expected_g_tp1
            )
            batch.q_target = g_target
            batch.q_tp1 = g_tp1

    # Prior probs are different on train and create_dist
    # def add_prior_probs(self, batch):
    #     self.memory.prior_dist = self.prior_policy()
    #     self.memory.prior_probs = sel

    def create_dist(self, state):
        g_values = self.forward(state)
        prior_dist = self.prior_policy(state)
        prior_probs = prior_dist.probs
        assert g_values.shape == prior_probs.shape

        # g_exp = g_values.exp()
        # prior_mult_g = prior_probs * g_exp
        # probs = prior_mult_g / prior_mult_g.sum()

        probs = F.softmax(prior_probs.log() + g_values)
        # self.add_histogram_log("Probs", probs)

        return tr.distributions.Categorical(probs=probs)

    def default_prior(self, state):
        """
        Create a uniform policy.
        """
        num_actions = self.batcher.get_action_info().shape
        probs = (1 / num_actions) * U.to_tensor(np.ones((state.shape[0], num_actions)))

        return tr.distributions.Categorical(probs=probs)

    @staticmethod
    def select_action(model, state, step, training=True):
        dist = model.create_dist(state)

        return U.to_np(dist.sample())
