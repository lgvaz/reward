from abc import ABC

from torchrl.utils import get_obj
from torchrl.utils.config import Config


class BaseAgent(ABC):
    '''
    Basic TorchRL agent. Encapsulate an environment and a model.

    Parameters
    ----------
    env: torchrl.envs
        A ``torchrl.envs`` instance.
    model: torchrl.models
        A ``torchrl.models`` instance.
    '''
    _model = None

    def __init__(self, env, model=None):
        self.env = env
        self.model = model or self._model

    def select_action(self, state):
        '''
        Receive a state and use the model to select an action.

        Parameters
        ----------
        state: numpy.ndarray
            The environment state.

        Returns
        -------
        action: int or numpy.ndarray
            The selected action.
        '''
        return self.model.select_action(state[None])

    def run_one_episode(self):
        '''
        Run an entire episode using the current model.

        Returns
        -------
        batch: dict
            Dictionary containing information about the episode.
        '''
        return self.env.run_one_episode(select_action_fn=self.select_action)

    @classmethod
    def from_config(cls, config):
        '''
        Create an agent from a configuration object.

        Returns
        -------
        torchrl.agents
            A TorchRL agent.
        '''
        env = get_obj(config.env.obj)
        state_shape = env.state_info['shape']
        action_shape = env.action_info['shape']
        model = cls._model.from_config(config, state_shape, action_shape)

        return cls(env, model)

    @classmethod
    def from_file(cls, file_path):
        '''
        Create an agent from a configuration file.

        Parameters
        ----------
        file_path: str
            Path to the configuration file.

        Returns
        -------
        torchrl.agents
            A TorchRL agent.
        '''
        config = Config.load(file_path)

        return cls.from_config(config)
