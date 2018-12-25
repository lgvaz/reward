import reward as rw, reward.utils as U
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, model, *, s_sp, a_sp):
        self.md, self.s_sp, self.a_sp = model, U.listify(s_sp), U.listify(a_sp)
        self._rsum, self._rs, self._eplen = None, [], None

    @abstractmethod
    def register_sa(self, s, a):
        for _ in range(s[0].shape[0]): U.global_step.add(1)
        s, a = U.listify(s), U.listify(a)
        self._check_s(s)
        self._check_a(a)

    @abstractmethod
    def report(self, r, d):
        assert r.shape == d.shape
        if self._rsum is None: self._rsum = r
        else:                  self._rsum += r
        if self._eplen is None: self._eplen = r * 0.
        self._eplen += 1
        self.write_ep_logs(d=d)

    def get_act(self, s):
        s = U.listify(s)
        self._check_s(s)
        st = [o.to_tensor() for o in s]
        a = [sp(U.to_np(o)) for o, sp in zip(U.listify(self.md.get_act(st)), self.a_sp)]
        self._check_a(a)
        self.register_sa(s=s, a=a)
        return a

    def write_ep_logs(self, d):
        for i in range(len(d)):
            if d[i]:
                self._rs.append(self._rsum[i])
                rw.logger.add_log('episode/reward', self._rsum[i], force=True)
                rw.logger.add_log('episode/len', self._eplen[i], force=True)
                self._rsum[i] = 0
                self._eplen[i] = 0
        rw.logger.add_header('Episode', len(self._rs))
    
    def _check_s(self, s): self._check_space(expected=self.s_sp, recv=s, name='State')
    def _check_a(self, a): self._check_space(expected=self.a_sp, recv=a, name='Action')
    
    def _check_space(self, expected, recv, name):
        if not len(expected) == len(recv): raise ValueError(f'Declared {name} has {len(expected)} inputs but received has {len(recv)}')
        for v1, v2 in zip(expected, recv):            
            if not hasattr(v2, 'sig'): raise TypeError(f'{name} must have a signature. (Image, Continuous or Categorical)')
            if not isinstance(v1, v2.sig): raise TypeError(f'{name} and Declared space dont match. Expected {v1} got {v2.sig}')