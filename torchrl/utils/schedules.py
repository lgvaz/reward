from functools import wraps
from torchrl.utils import to_np


def linear_schedule(initial_value, final_value, final_step, initial_step=0):
    """
    Calculates the value based on a linear interpolation.
    """
    decay_rate = -(initial_value - final_value) / (final_step - initial_step)

    @wraps(linear_schedule)
    def get(step):
        step = to_np(step)
        if step <= final_step:
            return decay_rate * (step - initial_step) + initial_value
        else:
            return final_value

    return get


def piecewise_linear_schedule(values, boundaries):
    """
    Junction of multiple linear interpolations.
    """
    boundaries = [0] + boundaries
    funcs = [
        linear_schedule(
            initial_value=iv, final_value=fv, final_step=fs, initial_step=is_
        )
        for iv, fv, fs, is_ in zip(
            values[:-1], values[1:], boundaries[1:], boundaries[:-1]
        )
    ]

    @wraps(piecewise_linear_schedule)
    def get(step):
        step = to_np(step)
        for i, bound in enumerate(boundaries[1:]):
            if step <= bound:
                return funcs[i](step)
        return funcs[-1](step)

    return get
