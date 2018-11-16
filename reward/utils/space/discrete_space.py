import numpy as np
from reward.utils.space import BaseSpace


class Discrete(BaseSpace):
    def __init__(self, num_acs):
        assert isinstance(num_acs, int)
        self.num_acs = num_acs
        super().__init__(shape=num_acs, dtype=np.int32)

    def __repr__(self):
        return "Discrete(num_actions={})".format(self.num_acs)

    def sample(self):
        return np.random.randint(low=0, high=self.num_acs, size=(1,))
