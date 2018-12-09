import numpy as np
import torch
import reward.utils as U
from .agent import Agent


class Replay(Agent):
    def __init__(self, model, *, s_sp, a_sp, bs, maxlen):
        super().__init__(model=model, s_sp=s_sp, a_sp=a_sp)
        self.bs = bs
        self.b = ReplayBuffer(maxlen=maxlen)
        
    def get_act(self, s):
        a = super().get_act(s=s)
        self.b.add_sa(s=U.listify(s), a=U.listify(a))
        return a
    
    def report(self, r, d):
        self.b.add_rd(r=r, d=d)
        if len(self.b) > self.bs:
            b = self.b.sample(bs=self.bs)            
            b['ss'] = [sp.from_list(o).to_tensor() for o, sp in zip(b['ss'], self.s_sp)]
            b['sns'] = [sp.from_list(o).to_tensor() for o, sp in zip(b['sns'], self.s_sp)]
            b['acs'] = [sp.from_list(o).to_tensor() for o, sp in zip(b['acs'], self.a_sp)]
            b['rs'] = torch.as_tensor(b['rs'], dtype=torch.float32, device=U.device.get_device())
            b['ds'] = torch.as_tensor(b['ds'], dtype=torch.float32, device=U.device.get_device())
            self.md.train(**b)


class ReplayBuffer:
    # TODO: Save and load
    def __init__(self, maxlen, num_envs=1):
        assert num_envs == 1, 'Only works with one env for now'
        self.maxlen = int(maxlen)
        self.buffer = []
        # Intialized at -1 so the first updated position is 0
        self.position = -1
        self._cycle = False

    def __len__(self): return len(self.buffer)

    def __getitem__(self, key): return self.buffer[key]

    def _get_batch(self, idxs):
        samples = [self[i] for i in idxs]
        b = U.memories.SimpleMemory.from_dicts(samples)
        # Add next state to batch
        b.sns = [self[i + 1]["ss"] for i in idxs]
        # Change shape from (#samples, #spaces) to (#spaces, #samples)
        b.update({k: list(zip(*b[k])) for k in ['ss', 'sns', 'acs']})
        return b
    
    def add_sa(self, s, a):
        if self._cycle: raise RuntimeError('add_sa and add_rd should be called sequentially')
        self._cycle = True
        # If buffer is not full, add a new element
        if len(self.buffer) < self.maxlen: self.buffer.append(None)
        self.position = (self.position + 1) % self.maxlen
        self.buffer[self.position] = dict(ss=s, acs=a)
    
    def add_rd(self, r, d):
        if not self._cycle: raise RuntimeError('add_sa and add_rd should be called sequentially')
        self._cycle = False
        self.buffer[self.position].update(dict(rs=r, ds=d))

    def sample(self, bs):
        idxs = np.random.choice(len(self) - 1, bs, replace=False)
        return self._get_batch(idxs=idxs)

    def save(self, savedir): raise NotImplementedError
    def load(self, loaddir): raise NotImplementedError