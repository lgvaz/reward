import PIL
import torch
import numpy as np
import reward.utils as U
import torchvision.transforms.functional as ttfm
from copy import deepcopy
from .space import Space

class Image(Space):
    def __init__(self, sz, order='chw'):
        assert order == 'chw', f'Only support order chw, got {order}'
        self.sz, self.order = sz, order

    def __call__(self, img): return ImageObj(img=img)
    def from_list(self, imgs): return ImageList(imgs=imgs)

# TODO: Suport for multiple envs
class ImageObj:
    sig = Image
    def __init__(self, img): self.img = img
    def __repr__(self): return f'Image({self.img.__repr__()})'

    @property
    def shape(self): raise NotImplementedError
    
    def to_tensor(self):
        # Hack for Stack
        try:                   x = self.img.to_pil()
        except AttributeError: x = self.img
        return ttfm.to_tensor(x)
    
    def apply_tfms(self, tfms):
        tfms = sorted(U.listify(tfms), key=lambda o: o.priority, reverse=True)
        x = self.clone()
        x.img = x if isinstance(x, PIL.Image.Image) else ttfm.to_pil_image(x.img)        
        for tfm in tfms: x.img = tfm(x.img)
        return x
    
    def clone(self): return self.__class__(img=deepcopy(self.img))

    @classmethod
    def from_list(cls, lst): return cls(np.array([o.img for o in lst]))

class ImageList:
    sig = Image
    def __init__(self, imgs): self.imgs = imgs
    def to_tensor(self): return torch.stack([o.to_tensor() for o in self.imgs])