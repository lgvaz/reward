import torch
import reward.utils as U
from .agent import Agent


class Rollout(Agent):
    def __init__(self, model, logger, *, s_sp, a_sp, bs):
        super().__init__(model=model, logger=logger, s_sp=s_sp, a_sp=a_sp)
        self.bs = bs
        self.b = RollBatch()
        
    def get_act(self, s):
        a = super().get_act(s=s)
        self.b.add_sa(s=U.listify(s), a=U.listify(a))
        return a
    
    def report(self, r, d):
        super().report(r=r, d=d)
        self.b.add_rd(r=r, d=d)
        if len(self.b) > self.bs: self.md.train(**self._get_batch())

    def _get_batch(self):
        b = self.b.get()            
        b['ss'] = [sp.from_list(o).to_tensor() for o, sp in zip(b['ss'], self.s_sp)]
        b['ss'], b['sns'] = [o[:-1] for o in b['ss']], [o[1:] for o in b['ss']]            
        b['acs'] = [sp.from_list(o).to_tensor() for o, sp in zip(b['acs'], self.a_sp)]
        b['rs'] = torch.as_tensor(b['rs'], dtype=torch.float32, device=U.device.get())
        b['ds'] = torch.as_tensor(b['ds'], dtype=torch.float32, device=U.device.get())
        return b


class RollBatch:
    def __init__(self): self.reset()        
        
    def __len__(self):
        return min(len(self.b.ss), len(self.b.rs))
        
    def add_sa(self, s, a):
        self._check()
        self.b.ss.append(s)
        self.b.acs.append(a)
        
    def add_rd(self, r, d):
        self.b.rs.append(r)
        self.b.ds.append(d)
        self._check()
        
    def get(self, reset=True):
        d = {k: v[:len(self.b.ss) - 1] for k, v in self.b.items()}
        # This includes all the states and next_states, so we only call to_tensor one time
        d['ss'].append(self.b.ss[-1])
        # Change shape from (#samples, #spaces) to (#spaces, #samples)
        d.update({k: list(zip(*d[k])) for k in ['ss', 'acs']})
        if reset: self.reset()
        return d
        
    def reset(self): self.b = U.memories.SimpleMemory(keys=['ss', 'acs', 'rs', 'ds'])
    
    def _check(self):
        if len(self.b.ss) != len(self.b.rs): raise RuntimeError('add_sa and add_rd should be called sequentially')