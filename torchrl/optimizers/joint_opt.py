import itertools
from torchrl.optimizers import BaseOpt
import torchrl.utils as U


class JointOpt(BaseOpt):
    @property
    def batch_keys(self):
        return list(itertools.chain(*[m.batch_keys for m in self.model]))

    @property
    def callbacks(self):
        return U.Callback.join_callbacks(*[m.callbacks for m in self.model])

    def model_parameters(self):
        return list(itertools.chain(*[m.parameters() for m in self.model]))

    def calculate_loss(self, batch):
        losses = [model.calculate_loss(batch) for model in self.model]
        loss = sum(losses)

        return loss
