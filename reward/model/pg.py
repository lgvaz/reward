from .model import Model
import reward.utils as U


class PG(Model):
    def __init__(self, policy, p_opt, gamma=0.99):
        self.p, self.p_opt, self.gamma = policy, p_opt, gamma
               
    def train(self, *, ss, sns, acs, rs, ds):
        ret = U.estim.disc_sum_rs(rs=rs, ds=ds, gamma=self.gamma)
        # TODO: Put this 2 lines on utils? (join_dims)
        ss, sns, acs = [[o.reshape((-1, *o.shape[2:])) for o in l] for l in [ss, sns, acs]]
        rs, ds, ret = [o.reshape((-1, *o.shape[2:])) for o in [rs, ds, ret]]
        dist, logprob = self.p.dist_logprob(ss=ss, acs=acs)
        assert ret.shape == logprob.shape
        loss = -(ret * logprob).mean()
        self.p_opt.zero_grad()
        loss.backward()
        self.p_opt.step()