import types


class BaseWrapper(object):
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def __str__(self):
        return "<{}{}>".format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    def __dir__(self):
        dir_list = super().__dir__()
        dir_list.extend(self.env.__dir__())

        return dir_list
