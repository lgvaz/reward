from torchrl.envs import GymEnv


class AtariEnv(GymEnv):
    @property
    def num_lives(self):
        return self.env.unwrapped.ale.lives()

    def get_action_meanings(self):
        return self.env.unwrapped.get_action_meanings()
