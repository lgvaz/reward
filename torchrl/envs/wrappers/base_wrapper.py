class BaseWrapper:
    def __new__(cls, env, **kwargs):
        '''
        Makes sure the FinalWrapper always stays last.
        '''
        self = super().__new__(cls)

        if env.__class__.__name__ == 'FinalWrapper':
            self.__init__(env.env, **kwargs)
            env.env = self
            return env

        else:
            self.__init__(env, **kwargs)
            return self

    def __init__(self, env):
        self.env = env
        self._shape = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    def __dir__(self):
        dir_list = super().__dir__()
        dir_list.extend(self.env.__dir__())

        return dir_list

    def get_state_info(self):
        info = self.env.get_state_info()

        if self._shape is None:
            self._shape = self.reset().shape

        info.shape = self._shape

        return info
