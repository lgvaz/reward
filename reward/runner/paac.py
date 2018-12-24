import warnings
import numpy as np
import torch
import torch.multiprocessing as mp
import reward.utils as U
from collections import namedtuple


def run(env_fn, n_envs, ss, acs, rs, ds, inq, outq, s_sp, tfms):
    envs = [env_fn() for _ in range(n_envs)]
    while True:
        signal = inq.get()
        a = acs.numpy()
        for i, env in enumerate(envs):
            if signal is None:
                s = np.array(s_sp(env.reset()[None]).apply_tfms(tfms))
                ss[i] = torch.as_tensor(s, dtype=ss.dtype)[0]
            else:
                # TODO: Squeeze may cause problems
                s, r, d, _ = env.step(a[i].squeeze())
                if d: s = env.reset()
                s = np.array(s_sp(s[None]).apply_tfms(tfms))
                ss[i] = torch.as_tensor(s, dtype=ss.dtype)[0]
                rs[i] = torch.as_tensor(r, dtype=rs.dtype)
                ds[i] = torch.as_tensor(d, dtype=ds.dtype)
        outq.put(None)


class PAAC:
    def __init__(self, env_fn, n_envs, s_sp, a_sp, n_workers=None, tfms=None):
        # TODO: Verify implemenatation, works with images? dtype with images, torch support uint8? Dont work with multiple spaces
        warnings.warn('Not tested with images')
        self.n_workers = n_workers or mp.cpu_count()
        if not n_envs % n_workers == 0 and n_envs > n_workers: raise ValueError('n_envs should be divisible by n_workers')
        self.env_fn,self.n_envs=env_fn,n_envs
        self._create_shared(s_sp=s_sp, a_sp=a_sp)
        self._create_workers(s_sp=s_sp, tfms=tfms)

    def reset(self):
        for w in self._workers: w.send.put(None)
        self._sync()
        return self._ss.clone()

    def step(self, act):
        self._acs.copy_(torch.as_tensor(act))
        for w in self._workers: w.send.put(True)
        self._sync()
        return self._ss.clone(), self._rs.clone(), self._ds.clone(), {}

    def close(self):
        for w in self._workers: w.p.terminate()

    def _sync(self):
        for w in self._workers: w.recv.get()

    def _create_shared(self, s_sp, a_sp):
        n_envs = (self.n_envs,)
        self._ss = torch.as_tensor(np.zeros(n_envs+tuple(s_sp.shape), dtype=s_sp.dtype))
        self._acs = torch.as_tensor(np.zeros(n_envs+tuple(a_sp.shape), dtype=a_sp.dtype))
        self._rs = torch.zeros(n_envs, dtype=torch.float)
        self._ds = torch.zeros(n_envs, dtype=torch.int)
        for t in [self._ss, self._acs, self._rs, self._ds]: t.share_memory_()

    def _create_workers(self, s_sp, tfms):
        Worker = namedtuple('Worker', 'p send recv')
        self._workers = []
        for ss, acs, rs, ds in zip(*self._split(self._ss, self._acs, self._rs, self._ds)):
            sendq, recvq = mp.Queue(), mp.Queue()
            p = mp.Process(target=run, args=(self.env_fn, self.n_envs//self.n_workers, ss, acs, rs, ds, sendq, recvq, s_sp, tfms))
            p.daemon = True
            p.start()
            self._workers.append(Worker(p=p, send=sendq, recv=recvq))

    def _split(self, *ts):
        # TODO: Asserting divisible, can simplify
        q, r = divmod(self.n_envs, self.n_workers)
        return [[t[i*q+min(i, r):(i+1)*q+min(i+1, r)] for i in range(self.n_workers)] for t in ts]





