# TODO: Deprecated
from torchrl.batchers.wrappers import BaseWrapper, StackFrames, Frame2Float


class ImageWrapper(BaseWrapper):
    def __init__(self, batcher, stack_frames=4):
        batcher = StackFrames(batcher=batcher, stack_frames=stack_frames)
        batcher = Frame2Float(batcher=batcher)
        self.batcher = batcher
        # super().__init__(batcher=batcher)
