import PIL
import torchvision.transforms.functional as ttfm
import reward.utils as U
from copy import deepcopy


class Image:
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