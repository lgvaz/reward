import torch
import reward.utils as U
from copy import deepcopy
from .space import Space


class Cont:
    space = Space
    def __init__(self, arr): self.arr = arr
        
    def __repr__(self): return f'Continuous({self.arr.__repr__()})'
        
    @property
    def shape(self): return self.arr.shape   
    
    def to_tensor(self):
        arr = arr.astype('float32') if U.is_np(arr) else arr
        return torch.as_tensor(arr, dtype=torch.float)

    def apply_tfms(self, tfms, priority=True):
        if priority: tfms = sorted(U.listify(tfms), key=lambda o: o.priority, reverse=True)
        x = self.clone()        
        for tfm in tfms: x.arr = tfm(x.arr)
        return x    
    
    def clone(self): return self.__class__(arr=deepcopy(self.arr))