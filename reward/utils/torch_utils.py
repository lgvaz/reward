import warnings
import torch
import numpy as np, reward as rw
from tqdm import tqdm
from pathlib import Path
from reward.utils import is_np, listify
from reward.utils.device import get

TDTYPE = dict(float=torch.float, float32=torch.float, double=torch.double, uint8=torch.uint8, int=torch.int, long=torch.long)

class OptimWrap:
    def __init__(self, opt, clip_grad_norm=None, callbacks=None):
        self.opt, self.callbacks = opt, listify(callbacks)
        self.clip_grad_norm = clip_grad_norm or float('inf')

    def optimize(self, loss, nn):
        self.zero_grad()
        loss.backward()
        for cb in self.callbacks: cb(nn.parameters())
        if self.clip_grad_norm != float('inf') or rw.logger.is_debug():
            gnorm = torch.nn.utils.clip_grad_norm_(nn.parameters(), self.clip_grad_norm)
            rw.logger.add_log(f'{nn.__class__.__name__}/grad_norm', gnorm, hidden=True)
        self.step()

    def step(self): return self.opt.step()
    def zero_grad(self): return self.opt.zero_grad()


# TODO: Deprecated
def to_tensor(x, dtype='float32', device=None):
    warnings.warn('to_tensor is probably going to be deprecated', DeprecationWarning, stacklevel=2)
    device = device or get()
    if is_np(x): x.astype(dtype)
    try:                   return x.to_tensor()
    except AttributeError: return torch.as_tensor(x, dtype=TDTYPE[dtype], device=device)

def optimize(loss, opt):
    opt.zero_grad()
    loss.backward()
    opt.step()

def copy_weights(from_nn, to_nn, weight):
    for fp, tp in zip(from_nn.parameters(), to_nn.parameters()):
        v = weight * fp.data + (1 - weight) * tp.data
        tp.data.copy_(v)

def mean_grad(nn):
    return torch.stack([p.grad.mean() for p in nn.parameters()]).mean()

def freeze_weights(nn):
    for param in nn.parameters(): param.requires_grad = False

def change_lr(opt, lr):
    for param_group in opt.param_groups: param_group["lr"] = lr

def save_model(model, save_dir, opt=None, step=0, is_best=False, name=None, postfix="checkpoint"):
    save_dir = Path(save_dir) / "models"
    save_dir.mkdir(exist_ok=True)
    name = name or model.__class__.__name__.lower()
    path = save_dir / "{}.pth.tar".format(name + "_" + postfix)
    save = {}

    save["model"] = model.state_dict()
    if opt is not None: save["opt"] = opt.state_dict()
    if step is not None: save["step"] = step
    torch.save(obj=save, f=str(path))
    if is_best:
        path = save_dir / "{}.pth.tar".format(name + "_best")
        torch.save(obj=save, f=str(path))

def load_model(model, path, opt=None):
    path = str(path) + ".pth.tar"
    load = torch.load(path)
    model.load_state_dict(load["model"])
    if opt is not None: opt.load_state_dict(load["opt"])
    tqdm.write("Loaded {} from {}".format(model.__class__.__name__, path))
