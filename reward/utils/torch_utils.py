import torch


def copy_weights(from_nn, to_nn, weight):
    for fp, tp in zip(from_nn.parameters(), to_nn.parameters()):
        v = weight * fp.data + (1 - weight) * tp.data
        tp.data.copy_(v)


def mean_grad(nn):
    return torch.stack([p.grad.mean() for p in nn.parameters()]).mean()
