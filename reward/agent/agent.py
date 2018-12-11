import reward.utils as U
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, model, logger, *, s_sp, a_sp):
        self.md, self.logger, self.s_sp, self.a_sp = model, logger, U.listify(s_sp), U.listify(a_sp)
        self._rsum, self._rs = None, []

    @abstractmethod
    def get_act(self, s):
        U.global_step.add(1)
        s = U.listify(s)
        self._check_s(s)
        s = [o.to_tensor() for o in s]
        a = [sp(U.to_np(o)) for o, sp in zip(U.listify(self.md.get_act(s)), self.a_sp)]
        self._check_a(a)
        return a
    
    @abstractmethod
    def report(self, r, d):
        assert r.shape == d.shape
        if self._rsum is None: self._rsum = r
        else:                  self._rsum += r
        for i in range(len(r)):
            if d[i]:
                self._rs.append(self._rsum[i])
                self.logger.add_log('reward', self._rsum[i], force=True)
                self._rsum[i] = 0
        self.logger.add_header('Episode', len(self._rs))

    
    def _check_s(self, s): self._check_space(expected=self.s_sp, recv=s, name='State')
    def _check_a(self, a): self._check_space(expected=self.a_sp, recv=a, name='Action')
    
    def _check_space(self, expected, recv, name):
        if not len(expected) == len(recv): raise ValueError(f'Declared {name} has {len(expected)} inputs but received has {len(recv)}')
        for v1, v2 in zip(expected, recv):            
            if not hasattr(v2, 'sig'): raise TypeError(f'{name} must have a signature. (Image, Continuous or Categorical)')
            if not isinstance(v1, v2.sig): raise TypeError(f'{name} and Declared space dont match. Expected {v1} got {v2.sig}')