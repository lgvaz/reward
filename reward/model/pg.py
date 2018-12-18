from .model import Model
import reward.utils as U


class PG(Model):
    def __init__(self, policy, p_opt, gamma=0.99):
        super().__init__(policy=policy)
        self.p_opt, self.gamma = self._wrap_opts(p_opt), gamma
               
    def train(self, *, ss, sns, acs, rs, ds):
        ret = U.estim.disc_sum_rs(rs=rs, ds=ds, gamma=self.gamma)
        # TODO: Put this 2 lines on utils? (join_dims)
        ss, sns, acs = [[o.reshape((-1, *o.shape[2:])) for o in l] for l in [ss, sns, acs]]
        rs, ds, ret = [o.reshape((-1, *o.shape[2:]))[..., None] for o in [rs, ds, ret]]
        dist = self.p.get_dist(*ss)
        logprob = self.p.logprob(dist, *acs)
        assert ret.shape == logprob.shape
        loss = -(ret * logprob).mean()
        self.p_opt.optimize(loss=loss, nn=self.p.nn)
        rw.logger.add_log("policy/loss", U.to_np(loss))