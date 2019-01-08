import torch
import reward.utils as U
from .agent import Agent
from reward.mem import ReplayBuffer


class Replay(Agent):
    def __init__(self, model, *, s_sp, a_sp, bs, maxlen, learn_freq=1., learn_start=0):
        super().__init__(model=model, s_sp=s_sp, a_sp=a_sp)
        self.bs, self.learn_freq, self.learn_start = bs, learn_freq, learn_start
        self.b = ReplayBuffer(maxlen=maxlen)
        
    def register_sa(self, s, a):
        super().register_sa(s=s, a=a)
        self.b.add_sa(s=U.listify(s), a=U.listify(a))

    def report(self, r, d):
        super().report(r=r, d=d)
        self.b.add_rd(r=r, d=d)
        gstep = U.global_step.get()
        if len(self.b) > self.bs and gstep % self.learn_freq == 0 and gstep > self.learn_start:
            self.md.train(**self._get_batch())

    def _get_batch(self):
        b = self.b.sample(bs=self.bs)            
        b['ss'] = [sp.from_list(o).to_tensor() for o, sp in zip(b['ss'], self.s_sp)]
        b['sns'] = [sp.from_list(o).to_tensor() for o, sp in zip(b['sns'], self.s_sp)]
        b['acs'] = [sp.from_list(o).to_tensor() for o, sp in zip(b['acs'], self.a_sp)]
        b['rs'] = U.tensor(b['rs'], dtype=torch.float32)
        b['ds'] = U.tensor(b['ds'], dtype=torch.float32)
        return b
