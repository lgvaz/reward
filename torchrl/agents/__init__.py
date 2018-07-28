# TODO: Explain more, after the implementation of more agents the vision will be more clear
"""
| The agent is the bridge between the model and the environment.
| It implements high level functions ready to be used by the user.
|
"""
from .base_agent import BaseAgent
from .pg_agent import PGAgent

__all__ = ["BaseAgent", "PGAgent"]
