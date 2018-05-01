from torchrl.envs import GymEnv


class RoboschoolEnv(GymEnv):
    def __init__(self, *args, **kwargs):
        import roboschool
        super().__init__(*args, **kwargs)
