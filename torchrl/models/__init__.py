'''
| A model encapsulate two PyTorch networks (body and head).
| It defines how actions are sampled from the network and a training procedure.
|
'''
from .base_model import BaseModel
from .pg_model import PGModel
from .reinforce_model import ReinforceModel
from .surrogate_pg import SurrogatePGModel

__all__ = ['BaseModel', 'PGModel', 'ReinforceModel', 'SurrogatePGModel']
