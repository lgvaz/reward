import itertools
from torchrl.utils.memories import SimpleMemory


class Callback:
    def __init__(self):
        self.callbacks = SimpleMemory(
            train_start=list(),
            epoch_start=list(),
            mini_batch_start=list(),
            train_end=list(),
            epoch_end=list(),
            mini_batch_end=list(),
            cleanup=list(),
        )

    def _run_callbacks(self, batch, callbacks):
        return any(c(batch) for c in callbacks)

    def _register_callback(self, func, callback):
        callback.append(func)

    def on_train_start(self, batch):
        return self._run_callbacks(batch=batch, callbacks=self.callbacks.train_start)

    def on_epoch_start(self, batch):
        return self._run_callbacks(batch=batch, callbacks=self.callbacks.epoch_start)

    def on_mini_batch_start(self, batch):
        return self._run_callbacks(
            batch=batch, callbacks=self.callbacks.mini_batch_start
        )

    def on_train_end(self, batch):
        return self._run_callbacks(batch=batch, callbacks=self.callbacks.train_end)

    def on_epoch_end(self, batch):
        return self._run_callbacks(batch=batch, callbacks=self.callbacks.epoch_end)

    def on_mini_batch_end(self, batch):
        return self._run_callbacks(batch=batch, callbacks=self.callbacks.mini_batch_end)

    def cleanups(self, batch):
        return self._run_callbacks(batch=batch, callbacks=self.callbacks.cleanup)

    def register_on_train_start(self, func):
        self._register_callback(func=func, callback=self.callbacks.train_start)

    def register_on_epoch_start(self, func):
        self._register_callback(func=func, callback=self.callbacks.epoch_start)

    def register_on_mini_batch_start(self, func):
        self._register_callback(func=func, callback=self.callbacks.mini_batch_start)

    def register_on_train_end(self, func):
        self._register_callback(func=func, callback=self.callbacks.train_end)

    def register_on_epoch_end(self, func):
        self._register_callback(func=func, callback=self.callbacks.epoch_end)

    def register_on_mini_batch_end(self, func):
        self._register_callback(func=func, callback=self.callbacks.mini_batch_end)

    def register_cleanup(self, func):
        self._register_callback(func=func, callback=self.callbacks.cleanup)

    @staticmethod
    def join_callbacks(*callbacks):
        new_callback = Callback()

        for k in new_callback.callbacks:
            new_callback.callbacks[k] = itertools.chain(
                *[c.callbacks[k] for c in callbacks]
            )

        return new_callback
