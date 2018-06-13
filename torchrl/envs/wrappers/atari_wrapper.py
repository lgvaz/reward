import torchrl.utils as U
from torchrl.envs.wrappers import (ActionRepeat, BaseWrapper, EpisodicLife, FireReset,
                                   RandomReset, StateWrapper)


class AtariWrapper(BaseWrapper):
    def __init__(self, env, shape=(84, 84), frame_skip=4, random_start_actions=30):
        env = EpisodicLife(env)
        env = RandomReset(env, num_actions=random_start_actions)
        env = ActionRepeat(env, skip=frame_skip)
        if 'FIRE' in env.get_action_meanings():
            env = FireReset(env)
        env = StateWrapper(
            env=env, funcs=[U.rgb_to_gray(),
                            U.rescale_img(shape),
                            U.hwc_to_chw()])
        # env = HWC_to_CHW(env)

        self.env = env
