from abc import ABC, abstractmethod
import reward.utils as U


class Model(ABC):
    @abstractmethod
    def train(self, *, ss, sns, acs, rs, ds): pass
    # TODO: carefull with shapes, probably want (num_samples, num_envs, feats)
            
    def get_act(self, ss): return self.p.get_act(*U.listify(ss))