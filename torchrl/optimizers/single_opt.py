from torchrl.optimizers import BaseOpt


class SingleOpt(BaseOpt):
    @property
    def batch_keys(self):
        return self.model.batch_keys

    @property
    def callbacks(self):
        return self.model.callbacks

    def model_parameters(self):
        return self.model.parameters()

    def calculate_loss(self, batch):
        return self.model.calculate_loss(batch)
