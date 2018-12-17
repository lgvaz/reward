import numpy as np
import torch, pickle
import reward.utils as U
from pathlib import Path
from .agent import Agent


class Replay(Agent):
    def __init__(self, model, logger, *, s_sp, a_sp, bs, maxlen, learn_freq=1., learn_start=0):
        super().__init__(model=model, logger=logger, s_sp=s_sp, a_sp=a_sp)
        self.bs, self.learn_freq, self.learn_start = bs, learn_freq, learn_start
        self.b = ReplayBuffer(maxlen=maxlen)
        
    def get_act(self, s):
        a = super().get_act(s=s)
        self.b.add_sa(s=U.listify(s), a=U.listify(a))
        return a
    
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
        b['rs'] = torch.as_tensor(b['rs'], dtype=torch.float32, device=U.device.get_device())
        b['ds'] = torch.as_tensor(b['ds'], dtype=torch.float32, device=U.device.get_device())
        return b


class ReplayBuffer:
    # TODO: Save and load
    def __init__(self, maxlen, num_envs=1):
        assert num_envs == 1, 'Only works with one env for now'
        # Position intialized at -1 so the first updated position is 0
        self.maxlen, self.buffer, self.position = int(maxlen), [], -1
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
        if len(self) < self.maxlen: self.buffer.append(None)
        self.position = (self.position + 1) % self.maxlen
        self.buffer[self.position] = dict(ss=s, acs=a)
    
    def add_rd(self, r, d):
        if not self._cycle: raise RuntimeError('add_sa and add_rd should be called sequentially')
        self._cycle = False
        self.buffer[self.position].update(dict(rs=r, ds=d))

    def add_transition(self, *, s, a, r, d):
        self.add_sa(s=s, a=a)
        self.add_rd(r=r, d=d)

    def sample(self, bs):
        idxs = np.random.choice(len(self) - 1, bs, replace=False)
        return self._get_batch(idxs=idxs)

    def save(self, savedir):
        # TODO: Not saving StackStates, include save option on spaces itself
        path = Path(savedir)/'buffer'
        path.mkdir(exist_ok=True, parents=True)
        mem = U.memories.SimpleMemory.from_dicts(self.buffer)
        mem.update({k: list(zip(*mem[k])) for k in ['ss', 'acs']})
        for i, s in enumerate(mem.ss): np.save(path/f'state_{i}.npy', np.array(s[0].from_list(s)))
        for i, a in enumerate(mem.acs): np.save(path/f'action_{i}.npy', np.array(a[0].from_list(a)))
        np.save(path/'reward.npy', np.array(mem.rs))
        np.save(path/'done.npy', np.array(mem.ds))
        info = {'s_sp': [o[0].__class__ for o in mem.ss], 'a_sp': [o[0].__class__ for o in mem.acs]}
        with open(str(path/'info.pkl'), 'wb') as f: pickle.dump(info, f)

    def load(self, loaddir):
        path = Path(loaddir)/'buffer'
        with open(str(path/'info.pkl'), 'rb') as f: info = pickle.load(f)
        ss,acs,rs,ds = self._load(path, 'state'),self._load(path, 'action'),self._load(path, 'reward'),self._load(path, 'done')
        ss, acs = self._load_space(arr=ss, sp=info['s_sp']), self._load_space(arr=acs, sp=info['a_sp'])
        ss, acs, rs, ds = list(zip(*ss)), list(zip(*acs)), rs[0], ds[0]
        assert len(ss) == len(acs) == len(rs) == len(ds)
        for s, a, r, d in zip(ss, acs, rs, ds): self.add_transition(s=s, a=a, r=r, d=d)

    def _load(self, path, name):
        items = sorted([str(p) for p in path.glob('*.npy') if name in str(p)])
        return [np.load(p) for p in items]

    def _load_space(self, arr, sp):
        return [[sp(x) for x in o] for o, sp in zip(arr, sp)]

