from torchrl.batchers.wrappers import *


class CommonWraps:
    @staticmethod
    def atari_wrap(batcher, stack_frames=4):
        batcher = CommonWraps.image_wrap(batcher, stack_frames=stack_frames)
        batcher = RewardClipper(batcher)
        return batcher

    @staticmethod
    def mujoco_wrap(batcher):
        batcher = StateRunNorm(batcher)
        batcher = RewardRunScaler(batcher)
        return batcher

    @staticmethod
    def image_wrap(batcher, stack_frames=4):
        batcher = StackFrames(batcher=batcher, stack_frames=stack_frames)
        batcher = Frame2Float(batcher=batcher)
        return batcher
