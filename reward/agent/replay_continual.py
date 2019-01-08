import torch
import reward.utils as U
from .replay import Replay
from reward.mem import DequeBuffer


class ReplayContinual(Replay):
    def __init__(self, model, *, s_sp, a_sp, bs, maxlen, on_split=.5, learn_freq=1., learn_start=0):
        super().__init__(model=model, s_sp=s_sp, a_sp=a_sp, bs=bs, maxlen=maxlen, learn_freq=1., learn_start=0)
        self.on_split = on_split
        self.onb = DequeBuffer(maxlen=int(bs*on_split))

    def register_sa(self, s, a):
        self.onb.add_sa(s=U.listify(s), a=U.listify(a))
        super().register_sa(s=s, a=a)

    def report(self, r, d):
        self.onb.add_rd(r=r, d=d)
        super().report(r=r, d=d)

    def _get_batch(self):
        b = self.b.sample(bs=int(self.bs * (1-self.on_split)))
        bon = self.onb.get()

        # TODO: This is ugly
        for i in range(len(b['acs'])): b['acs'][i].extend(bon['acs'][i])
        for i in range(len(b['ss'])): b['ss'][i].extend(bon['ss'][i])
        for i in range(len(b['sns'])): b['sns'][i].extend(bon['sns'][i])
        b['rs'].extend(bon['rs'])
        b['ds'].extend(bon['ds'])

        b['ss'] = [sp.from_list(o).to_tensor() for o, sp in zip(b['ss'], self.s_sp)]
        b['sns'] = [sp.from_list(o).to_tensor() for o, sp in zip(b['sns'], self.s_sp)]
        b['acs'] = [sp.from_list(o).to_tensor() for o, sp in zip(b['acs'], self.a_sp)]
        b['rs'] = U.tensor(b['rs'], dtype=torch.float32)
        b['ds'] = U.tensor(b['ds'], dtype=torch.float32)
        return b