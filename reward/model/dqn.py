import reward as rw, reward.utils as U
import torch.nn.functional as F
from .model import  Model


class DQN(Model):
    def __init__(self, *, policy, qnn, qnn_targ, q_opt, double=True, targ_up_w=1., targ_up_freq=10000, gamma=0.99):
        super().__init__(policy=policy)
        self.qnn,self.qnn_targ,self.q_opt = qnn,qnn_targ,self._wrap_opts(q_opt)
        self.double,self.targ_up_w,self.targ_up_freq,self.gamma = double,targ_up_w,targ_up_freq,gamma
        U.copy_weights(from_nn=self.qnn, to_nn=self.qnn_targ, weight=1.)
        U.global_step.subscribe_add(self._update_target_callback)

    def train(self, *, ss, sns, acs, rs, ds):
        # (#samples, #envs, #feats) -> (#samples + #envs, #feats)
        ss, sns, acs = [[o.reshape((-1, *o.shape[2:])) for o in l] for l in [ss, sns, acs]]
        rs, ds = [o.reshape((-1, *o.shape[2:]))[..., None] for o in [rs, ds]]
        if not len(acs) == 1: raise RuntimeError('Multi action space not suported')
        ### DQN update ###
        qb, qnb_targ = self.qnn(*ss), self.qnn_targ(*sns)
        if self.double: qnb_targ = qnb_targ.gather(dim=1, index=qb.argmax(dim=1, keepdim=True))
        else:           qnb_targ = qnb_targ.max(dim=1, keepdim=True)[0]
        select_qb = qb.gather(dim=1, index=acs[0][:, None])
        qtarg = U.estim.td_target(rs=rs, ds=ds, vn=qnb_targ, gamma=self.gamma).detach()
        loss = F.smooth_l1_loss(input=select_qb, target=qtarg)
        self.q_opt.optimize(loss=loss, nn=self.qnn)
        rw.logger.add_log('loss', loss, precision=4)
        rw.logger.add_log('q_mean', qb.mean(), hidden=True)
        rw.logger.add_histogram('acs', acs[0])
        rw.logger.add_histogram('q', select_qb)
        rw.logger.add_histogram('qtarg', qtarg)

    def _update_target_callback(self, gstep):
        if gstep % self.targ_up_freq == 0: U.copy_weights(from_nn=self.qnn, to_nn=self.qnn_targ, weight=self.targ_up_w)
