from reward.env import GymEnv


class AtariEnv(GymEnv):
    @property
    def num_lives(self):
        return self.env.unwrapped.ale.lives()

    def get_ac_meanings(self):
        return self.env.unwrapped.get_action_meanings()
