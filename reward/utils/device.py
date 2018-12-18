import torch

CONFIG = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}


def get(): return CONFIG["device"]
def set_device(device): CONFIG["device"] = device
