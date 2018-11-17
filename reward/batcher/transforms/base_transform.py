class BaseTransform:
    def transform_s(self, s, training=True):
        return s

    def transform_batch(self, batch, training=True):
        return batch

    def write_logs(self, logger):
        pass
