class BaseTransform:
    def transform_state(self, state, training=True):
        return state

    def transform_batch(self, batch, training=True):
        return batch

    def write_logs(self, logger):
        pass
