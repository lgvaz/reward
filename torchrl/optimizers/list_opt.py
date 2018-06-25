class ListOpt:
    def __init__(self, optimizers):
        self.opts = optimizers

    def learn_from_batch(self, batch, step):
        for opt in self.opts:
            opt.learn_from_batch(batch=batch, step=step)
