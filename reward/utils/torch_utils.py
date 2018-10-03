from tqdm import tqdm
import torch
from pathlib import Path


def copy_weights(from_nn, to_nn, weight):
    for fp, tp in zip(from_nn.parameters(), to_nn.parameters()):
        v = weight * fp.data + (1 - weight) * tp.data
        tp.data.copy_(v)


def mean_grad(nn):
    return torch.stack([p.grad.mean() for p in nn.parameters()]).mean()


def save_model(model, save_dir, opt=None, step=0, is_best=False, postfix="checkpoint"):
    save_dir = Path(save_dir) / "models"
    save_dir.mkdir(exist_ok=True)

    name = model.__class__.__name__.lower()
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
    path += ".pth.tar"
    load = torch.load(path)

    model.load_state_dict(load["model"])
    if opt is not None:
        opt.load_state_dict(load["opt"])

    name = model.__class__.__name__
    tqdm.write("Loaded {} from {}".format(name, path))
