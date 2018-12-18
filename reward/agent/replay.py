import numpy as np
import torch, pickle
import reward.utils as U
from pathlib import Path
from .agent import Agent


class Replay(Agent):
    def __init__(self, model, *, s_sp, a_sp, bs, maxlen, learn_freq=1., learn_start=0):
        super().__init__(model=model, s_sp=s_sp, a_sp=a_sp)
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
        b['rs'] = torch.as_tensor(b['rs'], dtype=torch.float32, device=U.device.get())
        b['ds'] = torch.as_tensor(b['ds'], dtype=torch.float32, device=U.device.get())
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
        path = Path(savedir)/'buffer'
        path.mkdir(exist_ok=True, parents=True)
        mem = U.memories.SimpleMemory.from_dicts(self.buffer)
        mem.update({k: list(zip(*mem[k])) for k in ['ss', 'acs']})
        info = {}
        info.update(self._save_space(mem.ss, name='state', savedir=path))
        info.update(self._save_space(mem.acs, name='action', savedir=path))
        np.save(path/'reward.npy', np.array(mem.rs))
        np.save(path/'done.npy', np.array(mem.ds))
        with open(str(path/'info.pkl'), 'wb') as f: pickle.dump(info, f)

    def load(self, loaddir):
        loaddir = Path(loaddir)/'buffer'
        with open(str(loaddir/'info.pkl'), 'rb') as f: info = pickle.load(f)
        ss = self._load_space(loaddir=loaddir, info=info['state'])
        acs = self._load_space(loaddir=loaddir, info=info['action'])
        rs, ds = np.load(loaddir/'reward.npy'), np.load(loaddir/'done.npy')
        assert len(ss) == len(acs) == len(rs) == len(ds)
        for s, a, r, d in zip(ss, acs, rs, ds): self.add_transition(s=s, a=a, r=r, d=d)

    def _save_space(self, x, name, savedir):
        info = {name: []}
        for i, o in enumerate(x):
            postfix = f'{name}_{i}'
            olist = o[0].from_list(o)
            olist.save(savedir=savedir, postfix=postfix)
            info[name].append(dict(name=postfix, cls=olist.__class__))
        return info

    def _load_space(self, loaddir, info):
        return list(zip(*[o['cls'].load(loaddir=loaddir, postfix=o['name']).unpack() for o in info]))