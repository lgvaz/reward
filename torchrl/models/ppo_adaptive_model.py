import torch
import torchrl.utils as U
from torchrl.models import SurrogatePGModel
from torch.distributions.kl import kl_divergence


class PPOAdaptiveModel(SurrogatePGModel):
    '''
    Proximal Policy Optimization as described in https://arxiv.org/pdf/1707.06347.pdf.


    Parameters
    ----------
    num_epochs: int
        How many times to train over the entire dataset (Default is 10).
    '''

    def __init__(self, model, env, kl_target=0.01, kl_penalty=1., num_epochs=10,
                 **kwargs):
        super().__init__(model=model, env=env, num_epochs=num_epochs, **kwargs)
        self.kl_target_fn = U.make_callable(kl_target)
        self.kl_penalty = kl_penalty

    @property
    def kl_target(self):
        return self.kl_target_fn(self.step)

    def register_losses(self):
        self.register_loss(self.surrogate_pg_loss)
        self.register_loss(self.kl_penalty_loss)
        self.register_loss(self.hinge_loss)
        self.register_loss(self.entropy_loss)

    def register_callbacks(self):
        super().register_callbacks()
        self.callbacks.register_on_epoch_end(self.kl_early_stopping)
        self.callbacks.register_on_train_end(self.kl_penalty_adjust)

    def kl_penalty_loss(self, batch):
        kl_div = kl_divergence(self.memory.old_dists,
                               self.memory.new_dists).sum(-1).mean()
        loss = self.kl_penalty * kl_div

        return loss

    def hinge_loss(self, batch):
        kl_div = kl_divergence(self.memory.old_dists,
                               self.memory.new_dists).sum(-1).mean()
        loss = 50 * max(0, kl_div - 2. * self.kl_target)**2

        return loss

    def kl_penalty_adjust(self, batch):
        # Adjust KL penalty
        if self.kl_div < self.kl_target / 1.5:
            self.kl_penalty /= 2
        if self.kl_div > self.kl_target * 1.5:
            self.kl_penalty *= 2

    def kl_early_stopping(self, batch):
        if self.kl_div > 4 * self.kl_target:
            print('Early stopping')
            return True

    def write_logs(self, batch):
        super().write_logs(batch)

        self.add_log('KL Target', self.kl_target, precision=4)
        self.add_log('KL Penalty', self.kl_penalty, precision=4)
