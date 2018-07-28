from torchrl.envs import GymEnv

try:
    import roboschool
except ImportError:
    pass  # soft dep


class RoboschoolEnv(GymEnv):
    """
    Support for gym Roboschool.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
