import torchrl.utils as U
from torchrl.envs.wrappers import BaseWrapper


class StateWrapper(BaseWrapper):
    def __init__(self, env, funcs=None):
        self.funcs = funcs or []
        # self.funcs.append(U.force_shape())
        super().__init__(env=env)

    def __str__(self):
        funcs_name = '-'.join(f.__name__ for f in self.funcs)
        name = '{}({})'.format(type(self).__name__, funcs_name)
        return '<{}{}>'.format(name, self.env)

    def reset(self):
        state = self.env.reset()
        for func in self.funcs:
            state = func(state)

        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        for func in self.funcs:
            state = func(state)
        return state, reward, done, info
