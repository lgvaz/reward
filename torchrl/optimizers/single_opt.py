from torchrl.optimizers import BaseOpt


class SingleOpt(BaseOpt):
    @property
    def name(self):
        return "/".join([self.__class__.__name__, self.model.name])

    @property
    def batch_keys(self):
        return self.model.batch_keys

    @property
    def callbacks(self):
        return self.model.callbacks

    @property
    def loss_coef(self):
        return self.loss_coef_sched(self.num_steps) or 1.

    def model_parameters(self):
        return self.model.parameters()

    def calculate_loss(self, batch):
        return self.model.calculate_loss(batch) * self.loss_coef
