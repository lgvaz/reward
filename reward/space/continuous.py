import torch
import numpy as np
import reward.utils as U
from copy import deepcopy
from .space import Space


class Continuous(Space):
    def __init__(self, low=None, high=None, shape=None):
        low, high = np.array(low), np.array(high)
        assert low.shape == high.shape if shape is None else True
        shape = shape or low.shape
        self.shape, self.dtype = shape, np.float32
        self.low = low + np.zeros(self.shape, dtype=self.dtype)
        self.high = high + np.zeros(self.shape, dtype=self.dtype)

    def __repr__(self): return f'Continuous(shape={self.shape},low={self.low},high={self.high})'

    def __call__(self, arr): return ContinuousObj(arr=arr)
    # def from_list(self, lst): return ContinuousObj.from_list(lst=lst)
    def from_list(self, arrs): return ContinuousList(arrs=arrs)

    def sample(self): return np.random.uniform(low=self.low, high=self.high, size=self.shape)

class ContinuousObj:
    sig = Continuous
    def __init__(self, arr): self.arr = np.array(arr, copy=False).astype('float32')
    def __repr__(self): return f'Continuous({self.arr.__repr__()})'
        
    @property
    def shape(self): return self.arr.shape   
    
    def to_tensor(self):
        return torch.as_tensor(self.arr, dtype=torch.float)

    def apply_tfms(self, tfms, priority=True):
        if priority: tfms = sorted(U.listify(tfms), key=lambda o: o.priority, reverse=True)
        x = self.clone()        
        for tfm in tfms: x.arr = tfm(x.arr)
        return x    
    
    def clone(self): return self.__class__(arr=deepcopy(self.arr))

class ContinuousList:
    sig = Continuous
    def __init__(self, arrs): self.arrs = arrs

    def to_tensor(self): return torch.as_tensor([o.arr for o in self.arrs], dtype=torch.float)