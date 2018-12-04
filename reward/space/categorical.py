import numpy as np
import reward.utils as U
from copy import deepcopy
from .space import Space


class Categorical(Space):
    def __init__(self, n_acs):
        assert isinstance(n_acs, int)
        self.n_acs, self.dtype = n_acs, np.int32

    def __repr__(self): return "Discrete(num_actions={})".format(self.n_acs)

    def __call__(self, val): return CategoricalObj(val=val)
    def from_list(self, vals): return CategoricalList(vals=vals)

    def sample(self): return np.random.randint(low=0, high=self.n_acs, size=(1,))


class CategoricalObj:
    sig = Categorical
    def __init__(self, val): self.val = val
    def __repr__(self): return f'Categorical({self.val.__repr__()})'

    @property
    def shape(self): raise NotImplementedError

    def to_tensor(self): return U.to_tensor(self.val)
    def apply_tfms(self, tfms, priority): raise NotImplementedError
    def clone(self): raise NotImplementedError

class CategoricalList:
    sig = Categorical
    def __init__(self, vals): self.vals = vals
    def __repr__(self): return f'Categorical({self.vals.__repr__()})'

    def to_tensor(self): return U.to_tensor([o.val for o in self.vals], dtype='int')