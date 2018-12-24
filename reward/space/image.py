import torch, json
import numpy as np, reward as rw, reward.utils as U
from .space import Space
from pathlib import Path
from reward.tfm.img.img import LazyStack


class Image(Space):
    def __init__(self, shape, order='NHWC', dtype=np.uint8):
        if order not in {'NHWC', 'NCHW'}: raise ValueError('Order must be NHWC or NCHW')
        # TODO: self.sz is not used
        self.shape, self.order, self.dtype = shape, order, dtype

    def __call__(self, img): 
        if len(img.shape) != 4: raise ValueError('Image should have 4 dimensions, NHWC or NCHW, specfied in constructor')
        if self.dtype != img.dtype: raise ValueError(f'Expected dtype {self.dtype} but got {img.dtype}')
        return ImageObj(img=self._fix_dims(img))

    def from_list(self, imgs): return ImageObj.from_list(imgs=imgs)

    def _fix_dims(self, img): return img if self.order == 'NHWC' else img.transpose([0, 2, 3, 1])


class ImageObj:
    sig = Image
    def __init__(self, img): self.img = img
    def __repr__(self): return f'Image({self.img.__repr__()})'
    
    def __array__(self):return np.array(self.img, copy=False)

    def to_tensor(self, transpose=True, device=None):
        arr = np.array(self).transpose([0, 3, 1, 2]) if transpose else np.array(self)
        x = U.tensor(arr, device=device)
        if isinstance(x, (torch.ByteTensor, torch.cuda.ByteTensor)): x = x.float() / 255.
        return x
    
    def apply_tfms(self, tfms):
        tfms = sorted(U.listify(tfms), key=lambda o: o.priority, reverse=True)
        img = self.img.copy()
        for tfm in tfms: img = tfm(img)
        return self.__class__(img=img)

    @staticmethod
    def from_list(imgs): return ImageList(imgs=imgs)

    @property
    def shape(self): raise NotImplementedError

class ImageList:
    sig = Image
    def __init__(self, imgs): self.imgs = imgs

    def __array__(self): 
        x = [o.img for o in self.imgs]
        # StackFrames Hack
        if isinstance(self.imgs[0].img, LazyStack): x = LazyStack.from_lists(x)
        return np.array(x)

    def to_tensor(self, transpose=True):
        arr = np.array(self).transpose([0, 1, 4, 2, 3]) if transpose else np.array(self)
        x = U.tensor(arr)
        if isinstance(x, (torch.ByteTensor, torch.cuda.ByteTensor)): x = x.float() / 255.
        return x

    def unpack(self): return self.imgs

    def save(self, savedir, postfix=''):
        savedir = Path(savedir)
        # StackFrames Hack
        if isinstance(self.imgs[0].img, LazyStack):
            x = np.array([np.array(o.img, copy=False)[..., -1, None] for o in self.imgs])
            with open(str(savedir/(f'lazystack_{postfix}.json')), 'w') as f: json.dump(dict(n=np.array(self.imgs[0]).shape[-1]), f)
        else:
             x = np.array([o.img for o in self.imgs])
        np.save(savedir/f'img_{postfix}.npy', x)

    @classmethod
    def load(cls, loaddir, postfix=''):
        arr = np.load(Path(loaddir)/f'img_{postfix}.npy')
        try:
            with open(str(Path(loaddir)/(f'lazystack_{postfix}.json')), 'r') as f: n = json.load(f)['n']
            stack = rw.tfm.img.Stack(n=n)
            arr = [stack(o) for o in arr]
        except FileNotFoundError: pass
        return cls([ImageObj(o) for o in arr])
        
