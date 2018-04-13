'''
| A model encapsulate two PyTorch networks (body and head).
| It defines how actions are sampled from the network and a training procedure.
|
'''
from .base_model import BaseModel
from .value_model import ValueModel
from .base_pg_model import BasePGModel
from .vanilla_pg_model import VanillaPGModel
from .surrogate_pg_model import SurrogatePGModel
from .ppo_model import PPOModel
from .pg_model import PGModel

__all__ = [
    'BaseModel', 'ValueModel', 'BasePGModel', 'VanillaPGModel', 'SurrogatePGModel',
    'PPOModel', 'PGModel'
]
