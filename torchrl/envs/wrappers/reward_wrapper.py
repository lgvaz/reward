from torchrl.envs.wrappers import BaseWrapper


class RewardWrapper(BaseWrapper):
    def __init__(self, env, funcs=None):
        self.funcs = funcs or []
        self.transformed_rewards = []
        super().__init__(env=env)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        for func in self.funcs:
            reward = func(reward)
        self.transformed_rewards = []

        return state, reward, done, info
