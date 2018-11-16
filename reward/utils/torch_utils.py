import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from reward.utils import to_np
from reward.utils.device import get_device


def to_tensor(x, cuda_default=True, force=False):
    if not isinstance(x, torch.Tensor):
        x = to_np(x)
        if x.dtype == "bool": x = x.astype("float32")
        if x.dtype == "float64": x = x.astype("float32")
        if x.dtype == "int": x = x.astype("float32")
        x = torch.from_numpy(x)

    return x.to(get_device())

def copy_weights(from_nn, to_nn, weight):
    for fp, tp in zip(from_nn.parameters(), to_nn.parameters()):
        v = weight * fp.data + (1 - weight) * tp.data
        tp.data.copy_(v)

def freeze_weights(nn):
    for param in nn.parameters():
        param.requires_grad = False

def mean_grad(nn):
    return torch.stack([p.grad.mean() for p in nn.parameters()]).mean()

def change_lr(opt, lr):
    for param_group in opt.param_groups:
        param_group["lr"] = lr

def save_model(
    model, save_dir, opt=None, step=0, is_best=False, name=None, postfix="checkpoint"
):
    save_dir = Path(save_dir) / "models"
    save_dir.mkdir(exist_ok=True)

    name = name or model.__class__.__name__.lower()
    path = save_dir / "{}.pth.tar".format(name + "_" + postfix)
    save = {}

    save["model"] = model.state_dict()
    if opt is not None:
        save["opt"] = opt.state_dict()
    if step is not None:
        save["step"] = step

    torch.save(obj=save, f=str(path))

    if is_best:
        path = save_dir / "{}.pth.tar".format(name + "_best")
        torch.save(obj=save, f=str(path))

def load_model(model, path, opt=None):
    path = str(path)
    path += ".pth.tar"
    load = torch.load(path)

    model.load_state_dict(load["model"])
    if opt is not None:
        opt.load_state_dict(load["opt"])

    name = model.__class__.__name__
    tqdm.write("Loaded {} from {}".format(name, path))
