import torch
import numpy as np
from reward.utils import to_tensor
from copy import deepcopy


def disc_sum_rs(rs, ds, vt_last=None, gamma=0.99):
    "Expected shape: (num_samples, num_envs)"
    rs = deepcopy(rs)

    # TODO: This works but is messy
    # TODO: I think tensor should be always be in CPU here (faster?)
    is_tensor = lambda x: isinstance(x, torch.Tensor)
    if any(map(is_tensor, (rs, ds, vt_last))):
        rs, ds = to_tensor(rs), to_tensor(ds)
        v_last = to_tensor(vt_last) if vt_last is not None else vt_last
        tensor = True
    else: tensor = False

    # if vt_last is None and not all(ds[-1]):
    #     raise AssertionError('If one episode is not finished you must'
    #                          'pass a value to vt_last to bootstrap from.')

    if vt_last is not None: rs[-1][ds[-1] == 0] = vt_last[ds[-1] == 0]

    returns = np.zeros(rs.shape, dtype="float32")
    returns_sum = np.zeros(rs.shape[-1], dtype="float32")
    if tensor: returns, returns_sum = map(to_tensor, (returns, returns_sum))

    for i in reversed(range(rs.shape[0])):
        returns_sum = rs[i] + gamma * returns_sum * (1 - ds[i])
        returns[i] = returns_sum

    return returns

def td_target(*, rs, ds, v_tp1, gamma):
    return rs + (1 - ds) * gamma * v_tp1

# TODO: Test this
def q_learning_target(*, rs, ds, q_tp1, gamma):
    max_q_tp1, _ = q_tp1.max(dim=1)
    return td_target(rs=rs, ds=ds, v_tp1=max_q_tp1, gamma=gamma)

def gae_estimation(rs, ds, v_t, v_tp1, *, gamma, gae_lambda):
    td_target_value = td_target(rs=rs, ds=ds, v_tp1=v_tp1, gamma=gamma)
    td_residual = td_target_value - v_t

    advantages = disc_sum_rs(rs=td_residual, ds=ds, vt_last=None, gamma=gamma * gae_lambda)

    return advantages
