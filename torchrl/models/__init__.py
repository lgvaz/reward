'''
| A model encapsulate two PyTorch networks (body and head).
| It defines how actions are sampled from the network and a training procedure.
'''
from .base_model import BaseModel
from .vanillapg_model import VanillaPGModel

__all__ = ['BaseModel', 'VanillaPGModel']
