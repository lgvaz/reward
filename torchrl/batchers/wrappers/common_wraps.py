from torchrl.batchers.wrappers import *


class CommonWraps:
    @staticmethod
    def image_wrap(batcher, stack_frames=4):
        batcher = StackFrames(batcher=batcher, stack_frames=stack_frames)
        batcher = Frame2Float(batcher=batcher)
        return batcher

    @staticmethod
    def mujoco_wrap(batcher):
        batcher = StateRunNorm(batcher)
        batcher = RewardRunScaler(batcher)
        return batcher
