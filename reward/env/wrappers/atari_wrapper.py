import reward.utils as U
from reward.env.wrappers import (
    ActionRepeat,
    BaseWrapper,
    EpisodicLife,
    FireReset,
    RandomReset,
    RGB2GRAY,
    Rescale,
    HWC2CHW,
    DelayedStart,
)


class AtariWrapper(BaseWrapper):
    def __init__(
        self, env, shape=(84, 84), frame_skip=4, random_start_acs=30, max_delay=1000
    ):
        env = EpisodicLife(env)
        env = DelayedStart(env, max_delay=max_delay)
        env = RandomReset(env, num_acs=random_start_acs)
        env = ActionRepeat(env, skip=frame_skip)
        if "FIRE" in env.get_ac_meanings():
            env = FireReset(env)
        env = RGB2GRAY(env)
        env = Rescale(env, shape=shape)
        env = HWC2CHW(env)

        self.env = env
