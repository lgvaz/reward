import torchrl.utils as U


class ListOpt:
    def __init__(self, optimizers):
        self.opts = optimizers

    def learn_from_batch(self, batch, step):
        # TODO: Currently always CUDA if possible (no choice)
        batch = batch.apply_to_all(U.to_tensor)
        for opt in self.opts:
            opt.learn_from_batch(batch=batch, step=step)

    def write_logs(self, logger):
        for opt in self.opts:
            opt.write_logs(logger=logger)
