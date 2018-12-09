from abc import ABC, abstractmethod
import reward.utils as U


class Model(ABC):
    def __init__(self, policy, logger):
        self.p, self.logger = policy, logger

    @abstractmethod
    def train(self, *, ss, sns, acs, rs, ds): pass
    # TODO: carefull with shapes, probably want (num_samples, num_envs, feats)
            
    def get_act(self, ss): return self.p.get_act(*U.listify(ss))