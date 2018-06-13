import numpy as np
from torchrl.envs.wrappers import BaseWrapper


class FinalWrapper(BaseWrapper):
    def __init__(self, env):
        # Don't hold any additional attributes, so it always get proxied to self.env
        self.env = env

    def reset(self):
        state = self.env.reset()
        return state[None]

    def step(self, action, auto_reset=True):
        state, reward, done, info = self.env.step(action)
        if auto_reset and done:
            state = self.env.reset()

        return state[None], np.array(reward)[None], np.array(done)[None], info

    # def get_state_info(self):
    #     info = self.env.get_state_info()
    #     info.shape = (1, ) + info.shape
    #     return info
