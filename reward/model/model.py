from abc import ABC, abstractmethod
import reward as rw, reward.utils as U


class Model(ABC):
    def __init__(self, policy, gamma):
        self.p, self.gamma = policy, gamma

    @abstractmethod
    def train(self, *, ss, sns, acs, rs, ds): pass
    # TODO: carefull with shapes, probably want (num_samples, num_envs, feats)
            
    def get_act(self, ss): return self.p.get_act(*U.listify(ss))

    def save_nn_callback(self, nn, opt, name=None):
        rw.logger.subscribe_log(self._save_nn_callback_fn(nn=nn, opt=opt, name=name))

    def _save_nn_callback_fn(self, nn, opt, name=None):
        def run():
            step, save_dir = U.global_step.get(), rw.logger.get_logdir()
            U.save_model(model=nn, save_dir=save_dir, opt=opt, step=step, name=name)
        return run

    def _wrap_opts(self, *opts):
        return U.delistify([opt if isinstance(opt, U.OptimWrap) else U.OptimWrap(opt) for opt in opts])