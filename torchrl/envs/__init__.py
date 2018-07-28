"""
The environment is the world that the agent interacts with, it could be a game,
a physics engine or anything you would like. It should receive and execute an
action and return to the agent the next observation and a reward.
"""

from .base_env import BaseEnv
from .gym_env import GymEnv
from .atari_env import AtariEnv
from .roboschool_env import RoboschoolEnv
from .osim_env import OsimRLEnv

__all__ = ["BaseEnv", "GymEnv", "RoboschoolEnv", "OsimRLEnv", "AtariEnv"]
