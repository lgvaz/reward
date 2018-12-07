import torch
import torch.nn.functional as F
import reward.utils as U
from .model import Model
from copy import deepcopy


class SAC(Model):
    def __init__(self, *, policy, q1nn, q2nn, vnn, vnn_targ, p_opt, q1_opt, q2_opt, v_opt, r_scale, vnn_targ_w=0.005, gamma=0.99):
        self.p,self.q1nn,self.q2nn,self.vnn,self.vnn_targ = policy,q1nn,q2nn,vnn,vnn_targ
        self.p_opt,self.q1_opt,self.q2_opt,self.v_opt = p_opt,q1_opt,q2_opt,v_opt
        self.r_scale,self.vnn_targ_w,self.gamma = r_scale,vnn_targ_w,gamma
        # Update value target nn
        U.copy_weights(from_nn=self.vnn, to_nn=self.vnn_targ, weight=1.)

    def train(self, *, ss, sns, acs, rs, ds):
        # TODO: Put this 2 lines on utils? (join_dims)
        ss, sns, acs = [[o.reshape((-1, *o.shape[2:])) for o in l] for l in [ss, sns, acs]]
        rs, ds = [o.reshape((-1, *o.shape[2:])) for o in [rs, ds]]

        q1b, q2b, vb = self.q1nn(*ss, *acs), self.q2nn(*ss, *acs), self.vnn(*ss)
        # TODO: Fix for multiple actions (dist.sample is wrong for multiple)
        dist = self.p.get_dist(*ss)
        anew, pretanh_anew = dist.rsample_with_pre()
        logprob = dist.log_prob_pre(pretanh_anew).sum(-1, keepdim=True)
        logprob /= float(self.r_scale)
        # Q loss
        vtarg_tp1 = self.vnn_targ(*sns)
        qt_next = U.estim.td_target(rs=rs, ds=ds, v_tp1=vtarg_tp1, gamma=self.gamma)
        q1_loss = (q1b - qt_next.detach()).pow(2).mean()
        q2_loss = (q2b - qt_next.detach()).pow(2).mean()
        # V Loss
        # TODO: Check anew, is it being passed correctly?
        q1new_t, q2new_t = self.q1nn(*ss, anew), self.q2nn(*ss, anew)
        qnew_t = torch.min(q1new_t, q2new_t)
        v_next = qnew_t - logprob
        v_loss = F.mse_loss(vb, v_next.detach())
        # Policy loss
        p_loss = (logprob - qnew_t).mean()
        # TODO: Add regulaziation losses
        # Optimize
        U.optimize(loss=q1_loss, opt=self.q1_opt)
        U.optimize(loss=q2_loss, opt=self.q2_opt)
        U.optimize(loss=v_loss, opt=self.v_opt)
        U.optimize(loss=p_loss, opt=self.p_opt)
        # Update value target nn
        U.copy_weights(from_nn=self.vnn, to_nn=self.vnn_targ, weight=self.vnn_targ_w)