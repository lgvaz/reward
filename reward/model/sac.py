import torch, torch.nn as nn, torch.nn.functional as F
import reward as rw, reward.utils as U
from copy import deepcopy
from .model import Model


class SAC(Model):
    def __init__(self, *, policy, q1nn, q2nn, p_opt, q1_opt, q2_opt, entropy=None, r_scale=1.0, targ_smooth=0.005, gamma=0.99):
        super().__init__(policy=policy, gamma=gamma)
        self.q1nn,self.q2nn,self.ent_targ,self.r_scale,self.targ_smooth = q1nn,q2nn,entropy,r_scale,targ_smooth
        self.p_opt,self.q1_opt,self.q2_opt = self._wrap_opts(p_opt,q1_opt,q2_opt)
        self.logtemp = nn.Parameter(torch.zeros(1).squeeze().to(U.device.get()))
        self.temp_opt = self._wrap_opts(torch.optim.Adam([self.logtemp], lr=self.p_opt.lr))
        self.q1nn_targ, self.q2nn_targ = deepcopy(q1nn).eval(), deepcopy(q2nn).eval()
        U.freeze_weights(self.q1nn_targ), U.freeze_weights(self.q2nn_targ)
        self._update_targ_nn(w=1.)
        self.save_nn_callback(nn=self.p.nn, opt=self.p_opt, name='pnn')
        self.save_nn_callback(nn=self.q1nn, opt=self.q1_opt, name='q1nn')
        self.save_nn_callback(nn=self.q2nn, opt=self.q2_opt, name='q2nn')

    def train(self, *, ss, sns, acs, rs, ds):
        # (#samples, #envs, #feats) -> (#samples + #envs, #feats)
        ss, sns, acs = [[o.reshape((-1, *o.shape[2:])) for o in l] for l in [ss, sns, acs]]
        rs, ds = [o.reshape((-1, *o.shape[2:]))[..., None] for o in [rs, ds]]
        ### Sac update ###
        q1b, q2b = self.q1nn(*ss, *acs), self.q2nn(*ss, *acs)
        dist, distn = self.p.get_dist(*ss), self.p.get_dist(*sns)
        anew, anew_pre = map(U.listify, self.p.get_act_pre(dist=dist))
        anewn, anew_pren = map(U.listify, self.p.get_act_pre(dist=distn))
        logprob = self.p.logprob_pre(dist, *anew_pre) / float(self.r_scale)
        logprobn = self.p.logprob_pre(distn, *anew_pren) / float(self.r_scale)
        assert logprob.shape == logprobn.shape == q1b.shape == q2b.shape == rs.shape == ds.shape
        # Q loss
        q1targn, q2targn = self.q1nn_targ(*sns, *anewn), self.q2nn_targ(*sns, *anewn)
        qtargn = torch.min(q1targn, q2targn) - self.temp.detach() * logprobn
        q_tdtarg = U.estim.td_target(rs=rs, ds=ds, vn=qtargn, gamma=self.gamma)
        q1_loss = (q1b - q_tdtarg.detach()).pow(2).mean()
        q2_loss = (q2b - q_tdtarg.detach()).pow(2).mean()
        # Policy loss
        q1new, q2new = self.q1nn(*ss, *anew), self.q2nn(*ss, *anew)
        qnew = torch.min(q1new, q2new)
        p_loss = (self.temp.detach() * logprob - qnew).mean() 
        p_loss += 1e-3 * self.p.mean(dist=dist).pow(2).mean() + 1e-3 * self.p.std(dist=dist).log().pow(2).mean()
        # Optimize
        self.q1_opt.optimize(loss=q1_loss, nn=self.q1nn)
        self.q2_opt.optimize(loss=q2_loss, nn=self.q2nn)
        self.p_opt.optimize(loss=p_loss, nn=self.p.nn)
        self._update_targ_nn(w=self.targ_smooth)
        if self.ent_targ is not None:
            temp_loss = -self.logtemp * (logprob + self.ent_targ).detach().mean()
            self.temp_opt.optimize(loss=temp_loss)
        # Write logs
        rw.logger.add_log("policy/loss", p_loss)
        rw.logger.add_log("policy/logprob_mean", logprob.mean(), hidden=False)
        rw.logger.add_log("q1/loss", q1_loss)
        rw.logger.add_log("q2/loss", q2_loss)
        rw.logger.add_log('temperature', self.temp, precision=4)
        rw.logger.add_histogram("policy/logprob", logprob)
        rw.logger.add_histogram("policy/mean", self.p.mean(dist=dist))
        rw.logger.add_histogram("policy/std", self.p.std(dist=dist))
        rw.logger.add_histogram("q1/value", q1b)
        rw.logger.add_histogram("q2/value", q2b)

    @property
    def temp(self): return self.logtemp.exp()

    def _update_targ_nn(self, w):
        U.copy_weights(from_nn=self.q1nn, to_nn=self.q1nn_targ, weight=w)
        U.copy_weights(from_nn=self.q2nn, to_nn=self.q2nn_targ, weight=w)
