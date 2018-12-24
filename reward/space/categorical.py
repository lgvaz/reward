import torch
import numpy as np
import reward.utils as U
from pathlib import Path
from .space import Space


class Categorical(Space):
    def __init__(self, n_acs):
        assert isinstance(n_acs, int)
        self.n_acs, self.dtype = n_acs, np.int32

    def __repr__(self): return "Discrete(num_actions={})".format(self.n_acs)

    def __call__(self, val): return CategoricalObj(val=val)
    def from_list(self, vals): return CategoricalObj.from_list(vals=vals)

    def sample(self): return np.random.randint(low=0, high=self.n_acs, size=(1,))

    @property
    def shape(self): return (1,)


class CategoricalObj:
    sig = Categorical
    def __init__(self, val): self.val = val
    def __repr__(self): return f'Categorical({self.val.__repr__()})'

    def __array__(self): return np.array(val, dtype='int', copy=False)
    def to_tensor(self): return U.tensor(np.array(self))

    def apply_tfms(self, tfms, priority): raise NotImplementedError

    @staticmethod
    def from_list(vals): return CategoricalList(vals=vals)

    @property
    def shape(self): raise NotImplementedError


class CategoricalList:
    sig = Categorical
    def __init__(self, vals): self.vals = vals
    def __repr__(self): return f'Categorical({self.vals.__repr__()})'

    def __array__(self): return np.array([o.val for o in self.vals], dtype='int', copy=False)
    def to_tensor(self): return U.tensor(np.array(self))

    def unpack(self): return self.vals

    def save(self, savedir, postfix=''):
        np.save(Path(savedir)/f'cat_{postfix}.npy', np.array(self))

    @classmethod
    def load(cls, loaddir, postfix=''):
        arr = np.load(Path(loaddir)/f'cat_{postfix}.npy')
        return cls([CategoricalObj(o) for o in arr])
        
