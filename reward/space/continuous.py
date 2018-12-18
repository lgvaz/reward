import torch
import numpy as np, reward.utils as U
from pathlib import Path
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
    def from_list(self, arrs): return ContinuousObj.from_list(arrs=arrs)

    def sample(self): return np.random.uniform(low=self.low, high=self.high, size=self.shape)

class ContinuousObj:
    sig = Continuous
    def __init__(self, arr): self.arr = np.array(arr, dtype='float', copy=False)
    def __repr__(self): return f'Continuous({self.arr.__repr__()})'
        
    @property
    def shape(self): return self.arr.shape   
    
    def __array__(self): return np.array(self.arr, dtype='float', copy=False)
    def to_tensor(self): return torch.as_tensor(np.array(self), dtype=torch.float, device=U.device.get())

    def apply_tfms(self, tfms, priority=True):
        if priority: tfms = sorted(U.listify(tfms), key=lambda o: o.priority, reverse=True)
        x = self.arr.copy()
        for tfm in tfms: x = tfm(x)
        return self.__class__(arr=x)

    @staticmethod
    def from_list(arrs): return ContinuousList(arrs=arrs)


class ContinuousList:
    sig = Continuous
    def __init__(self, arrs): self.arrs = arrs

    def __array__(self): return np.array([o.arr for o in self.arrs], dtype='float', copy=False)
    def to_tensor(self): return torch.as_tensor(np.array(self), dtype=torch.float, device=U.device.get())

    def unpack(self): return self.arrs

    def save(self, savedir, postfix=''):
        np.save(Path(savedir)/f'cont_{postfix}.npy', np.array(self))

    @classmethod
    def load(cls, loaddir, postfix=''):
        arr = np.load(Path(loaddir)/f'cont_{postfix}.npy')
        return cls([ContinuousObj(o) for o in arr])
        
