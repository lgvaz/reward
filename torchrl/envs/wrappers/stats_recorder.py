import numpy as np
from torchrl.envs.wrappers import BaseWrapper


# TODO: Is this wrapper needed?
class StatsRecorder(BaseWrapper):
    def __init__(self, env):
        self.real_num_steps = 0
        self.real_num_episodes = 0
        self.ep_reward_sum = 0
        self.rewards = []
        super().__init__(env=env)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.ep_reward_sum += reward

        if done:
            self.rewards.append(self.ep_reward_sum)
            self.ep_reward_sum = 0
            self.real_num_episodes += 1

        self.real_num_steps += 1

        return state, reward, done, info

    def write_logs(self, logger):
        self.env.write_logs(logger)
        logger.add_log("Env/Reward/Episode (Last 50)", np.mean(self.rewards[-50:]))
