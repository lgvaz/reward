import reward as rw
from abc import ABC, abstractmethod
from functools import wraps
from reward.utils import to_np, global_step


class Sched(ABC):
    @abstractmethod
    def get(self, step): pass

    def __call__(self, step): 
        val = self.get(step)
        rw.logger.add_log(f'Scheds/{self.__class__.__name__}', val, precision=3)
        return val

class Linear(Sched):
    "Calculates the value based on a linear interpolation."
    def __init__(self, initial_value, final_value, final_step, initial_step=0):
        self.initv,self.finalv,self.finals,self.inits = initial_value, final_value, final_step, initial_step
        self.decay = -(initial_value - final_value) / (final_step - initial_step)

    def get(self, step):
        if step <= self.finals: return float(self.decay * (step - self.inits) + self.initv)
        else:                   return float(self.finalv)

class PieceLinear(Sched):
    "Junction of multiple linear interpolations."
    def __init__(self, values, bounds):
        self.bounds = [0] + bounds
        self.funcs = [Linear(iv, fv, fs, is_) for iv, fv, fs, is_ in zip(values[:-1], values[1:], self.bounds[1:], self.bounds[:-1])]

    def get(self, step):
        for i, bound in enumerate(self.bounds[1:]):
            if step <= bound: return self.funcs[i].get(step)
        return self.funcs[-1].get(step)

class PieceConst(Sched):
    def __init__(self, values, bounds):
        self.values, self.bounds = values, bounds

    def get(self, step):
        for i, bound in enumerate(self.bounds):
            if step <= bound: return self.values[i]
        return self.values[-1]