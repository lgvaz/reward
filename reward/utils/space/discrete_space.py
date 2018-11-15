import numpy as np
from reward.utils.space import BaseSpace


class Discrete(BaseSpace):
    def __init__(self, num_actions):
        assert isinstance(num_actions, int)
        self.num_actions = num_actions
        super().__init__(shape=num_actions, dtype=np.int32)

    def __repr__(self):
        return "Discrete(num_actions={})".format(self.num_actions)

    def sample(self):
        return np.random.randint(low=0, high=self.num_actions, size=(1,))
