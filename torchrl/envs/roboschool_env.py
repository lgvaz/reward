from torchrl.envs import GymEnv

try:
    import roboschool
except ModuleNotFoundError:
    pass


class RoboschoolEnv(GymEnv):
    pass
