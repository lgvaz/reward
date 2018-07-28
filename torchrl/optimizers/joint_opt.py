import itertools
from torchrl.optimizers import BaseOpt
import torchrl.utils as U


class JointOpt(BaseOpt):
    @property
    def name(self):
        models_name = "-".join([m.name for m in self.model])
        return "/".join([self.__class__.__name__, models_name])

    @property
    def batch_keys(self):
        return list(set(itertools.chain(*[m.batch_keys for m in self.model])))

    @property
    def callbacks(self):
        return U.Callback.join_callbacks(*[m.callbacks for m in self.model])

    @property
    def loss_coef(self):
        try:
            return [coef(self.num_steps) for coef in self.loss_coef_sched]
        except TypeError:
            return [1.] * len(self.model)

    def model_parameters(self):
        return list(itertools.chain(*[m.parameters() for m in self.model]))

    def calculate_loss(self, batch):
        losses = [model.calculate_loss(batch) for model in self.model]
        loss = sum([l * coef for l, coef in zip(losses, self.loss_coef)])

        return loss
