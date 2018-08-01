from torchrl.batchers.transforms import *


def atari_transforms(stack_frames=4):
    return [StackStates(n=stack_frames), Frame2Float(), RewardClipper()]


def mujoco_transforms():
    return [StateRunNorm(), RewardRunScaler()]
