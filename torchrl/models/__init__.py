'''
| A model encapsulate two PyTorch networks (body and head).
| It defines how actions are sampled from the network and a training procedure.
|
'''
from .base_model import BaseModel
from .base_pg_model import BasePGModel
from .reinforce_model import ReinforceModel
from .surrogate_pg_model import SurrogatePGModel
from .ppo_model import PPOModel

__all__ = ['BaseModel', 'BasePGModel', 'ReinforceModel', 'SurrogatePGModel', 'PPOModel']
