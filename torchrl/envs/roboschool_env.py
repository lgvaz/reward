from torchrl.envs import GymEnv


class RoboschoolEnv(GymEnv):
    '''
    Support for gym Roboschool.
    '''

    def __init__(self, *args, **kwargs):
        import roboschool
        super().__init__(*args, **kwargs)
