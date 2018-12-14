import PIL
import torch
import numpy as np
import reward.utils as U
import torchvision.transforms.functional as ttfm
from copy import deepcopy
from .space import Space
from reward.tfm.img.img import LazyStack


class Image(Space):
    def __init__(self, sz, order='NHWC'):
        if order not in {'NHWC', 'NCHW'}: raise ValueError('Order must be NHWC or NCHW')
        # TODO: self.sz is not used
        self.sz, self.order = sz, order

    def __call__(self, img): 
        if len(img.shape) != 4: raise ValueError('Image should have 4 dimensions, NHWC or NCHW, specfied in constructor')
        return ImageObj(img=self._fix_dims(img))
    def from_list(self, imgs): return ImageList(imgs=imgs)

    def _fix_dims(self, img): return img if self.order == 'NHWC' else img.transpose([0, 2, 3, 1])

class ImageObj:
    sig = Image
    def __init__(self, img): self.img = img
    def __repr__(self): return f'Image({self.img.__repr__()})'
    
    def __array__(self):return np.array(self.img).transpose([0, 3, 1, 2])

    @property
    def shape(self): raise NotImplementedError
    
    def to_tensor(self):
        x = torch.as_tensor(np.array(self), device=U.device.get_device())
        if isinstance(x, (torch.ByteTensor, torch.cuda.ByteTensor)): x = x.float() / 255.
        return x
    
    def apply_tfms(self, tfms):
        tfms = sorted(U.listify(tfms), key=lambda o: o.priority, reverse=True)
        x = self.clone()
        for tfm in tfms: x.img = tfm(x.img)
        return x
    
    def clone(self): return self.__class__(img=deepcopy(self.img))

    @classmethod
    def from_list(cls, lst): return cls(np.array([o.img for o in lst]))

class ImageList:
    sig = Image
    def __init__(self, imgs): self.imgs = imgs

    def __array__(self): 
        # StackFrames Hack
        imgs = [np.array(img.img) if isinstance(img.img, LazyStack) else img.img for img in self.imgs]
        return np.array(imgs).transpose([0, 1, 4, 2, 3])

    def to_tensor(self):
        x = torch.as_tensor(np.array(self), device=U.device.get_device())
        if isinstance(x, (torch.ByteTensor, torch.cuda.ByteTensor)): x = x.float() / 255.
        return x