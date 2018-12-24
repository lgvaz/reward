import torch
import torch.nn.functional as F
import reward as rw, reward.utils as U
from .model import Model


class SAC(Model):
    def __init__(self, *, policy, q1nn, q2nn, vnn, vnn_targ, p_opt, q1_opt, q2_opt, v_opt, r_scale, vnn_targ_w=0.005, gamma=0.99):
        super().__init__(policy=policy)
        self.q1nn,self.q2nn,self.vnn,self.vnn_targ = q1nn,q2nn,vnn,vnn_targ
        self.p_opt,self.q1_opt,self.q2_opt,self.v_opt = self._wrap_opts(p_opt,q1_opt,q2_opt,v_opt)
        self.r_scale,self.vnn_targ_w,self.gamma = r_scale,vnn_targ_w,gamma
        # Update value target nn
        U.copy_weights(from_nn=self.vnn, to_nn=self.vnn_targ, weight=1.)

    def train(self, *, ss, sns, acs, rs, ds):
        # (#samples, #envs, #feats) -> (#samples + #envs, #feats)
        ss, sns, acs = [[o.reshape((-1, *o.shape[2:])) for o in l] for l in [ss, sns, acs]]
        rs, ds = [o.reshape((-1, *o.shape[2:]))[..., None] for o in [rs, ds]]
        ### Sac update ###
        q1b, q2b, vb = self.q1nn(*ss, *acs), self.q2nn(*ss, *acs), self.vnn(*ss)
        dist = self.p.get_dist(*ss)
        anew, anew_pre = map(U.listify, self.p.get_act_pre(dist=dist))
        # TODO: Test scale reward
        logprob = self.p.logprob_pre(dist, *anew_pre) / float(self.r_scale)
        assert logprob.shape == q1b.shape == q2b.shape == vb.shape == rs.shape == ds.shape
        # Q loss
        vtarg_tp1 = self.vnn_targ(*sns)
        qt_next = U.estim.td_target(rs=rs, ds=ds, vn=vtarg_tp1, gamma=self.gamma)
        q1_loss = (q1b - qt_next.detach()).pow(2).mean()
        q2_loss = (q2b - qt_next.detach()).pow(2).mean()
        # V Loss
        q1new_t, q2new_t = self.q1nn(*ss, *anew), self.q2nn(*ss, *anew)
        qnew_t = torch.min(q1new_t, q2new_t)
        v_next = qnew_t - logprob
        v_loss = F.mse_loss(vb, v_next.detach())
        # Policy loss
        p_loss = (logprob - qnew_t).mean() 
        p_loss += 1e-3 * self.p.mean(dist=dist).pow(2).mean() + 1e-3 * self.p.std(dist=dist).log().pow(2).mean()
        # Optimize
        self.q1_opt.optimize(loss=q1_loss, nn=self.q1nn)
        self.q2_opt.optimize(loss=q2_loss, nn=self.q2nn)
        self.v_opt.optimize(loss=v_loss, nn=self.vnn)
        self.p_opt.optimize(loss=p_loss, nn=self.p.nn)
        # Update value target nn
        U.copy_weights(from_nn=self.vnn, to_nn=self.vnn_targ, weight=self.vnn_targ_w)
        # Write logs
        rw.logger.add_log("policy/loss", p_loss)
        rw.logger.add_log("v/loss", v_loss)
        rw.logger.add_log("q1/loss", q1_loss)
        rw.logger.add_log("q2/loss", q2_loss)
        rw.logger.add_histogram("policy/logprob", logprob)
        rw.logger.add_histogram("policy/mean", self.p.mean(dist=dist))
        rw.logger.add_histogram("policy/std", self.p.std(dist=dist))
        rw.logger.add_histogram("v/value", vb)
        rw.logger.add_histogram("q1/value", q1b)
        rw.logger.add_histogram("q2/value", q2b)

