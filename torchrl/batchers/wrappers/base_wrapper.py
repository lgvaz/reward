import inspect


# TODO: functools.wraps new for parameters
class BaseWrapper:
    def __new__(cls, batcher, **kwargs):
        self = super().__new__(cls)
        self.__init__(batcher, **kwargs)

        # Automatically points unwrapped methods to last wrapper
        for attr in dir(batcher):
            # Don't get magic methods
            if attr.startswith("__") or attr.startswith("old"):
                continue

            value = getattr(batcher, attr)
            if inspect.ismethod(value):
                # Store old method
                setattr(self, "old_" + attr, value)
                # Points to new method
                setattr(batcher.unwrapped, attr, getattr(self, attr))

        return self

    def __init__(self, batcher):
        self.batcher = batcher

    def __str__(self):
        return "<{}{}>".format(type(self).__name__, self.batcher)

    def __repr__(self):
        return str(self)

    # Delegate all non-implemented attrs calls to wrapped class
    def __getattr__(self, name):
        return getattr(self.batcher, name)

    # Helps with editor auto-completion
    def __dir__(self):
        dir_list = super().__dir__()
        dir_list.extend(self.batcher.__dir__())

        return dir_list
