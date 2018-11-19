from .space import Space


# TODO: Probably deprecated
class MultiSpace(Space):
    def __init__(self, *spaces):
        self.spaces = spaces

    def __repr__(self):
        return ", ".join([sp.__repr__() for sp in self.spaces])

    def sample(self):
        return [sp.sample() for sp in self.spaces]

