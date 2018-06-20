import torchrl.utils as U
from torchrl.envs.wrappers import (ActionRepeat, BaseWrapper, EpisodicLife, FireReset,
                                   RandomReset, RGB2GRAY, Rescale, HWC2CHW)


class AtariWrapper(BaseWrapper):
    def __init__(self, env, shape=(84, 84), frame_skip=4, random_start_actions=30):
        env = EpisodicLife(env)
        env = RandomReset(env, num_actions=random_start_actions)
        env = ActionRepeat(env, skip=frame_skip)
        if 'FIRE' in env.get_action_meanings():
            env = FireReset(env)
        # TODO: I think it better to have separte classes for the transforms
        # env = StateWrapper(
        #     env=env, funcs=[U.rgb_to_gray(),
        #                     U.rescale_img(shape),
        #                     U.hwc_to_chw()])
        env = RGB2GRAY(env)
        env = Rescale(env, shape=shape)
        env = HWC2CHW(env)

        self.env = env
