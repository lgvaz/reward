from torchrl.envs import GymEnv

try:
    from osim.env import *
except ImportError:
    _has_osim = False
else:
    _has_osim = True


class OsimRLEnv(GymEnv):
    def __init__(self, env_name, visualize=False, **kwargs):
        if not _has_osim:
            raise ImportError("Osim is required to use this class")

        self.visualize = visualize
        super(GymEnv, self).__init__(env_name, **kwargs)

    def _create_env(self):
        env = eval(self.env_name)(visualize=self.visualize)

        return env
