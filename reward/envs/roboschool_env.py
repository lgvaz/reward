from reward.envs import GymEnv

# Soft dependency
try:
    import roboschool
except ImportError:
    _has_roboschool = False
else:
    _has_roboschool = True


class RoboschoolEnv(GymEnv):
    """
    Support for gym Roboschool.
    """

    def __init__(self, *args, **kwargs):
        if not _has_roboschool:
            raise ImportError("Could not import roboschool")
        super().__init__(*args, **kwargs)
