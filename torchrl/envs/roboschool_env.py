from torchrl.envs import GymEnv


class RoboschoolEnv(GymEnv):
    '''
    Support for gym Roboschool.
    '''

    # TODO: Test roboschool import
    def __init__(self, *args, **kwargs):
        import roboschool
        super().__init__(*args, **kwargs)
