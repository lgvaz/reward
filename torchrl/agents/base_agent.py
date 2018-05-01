import numpy as np
from abc import ABC, abstractmethod

import torchrl.utils as U


class BaseAgent(ABC):
    '''
    Basic TorchRL agent. Encapsulate an environment and a model.

    Parameters
    ----------
    env: torchrl.envs
        A ``torchrl.envs`` instance.
    gamma: float
        Discount factor on future rewards (Default is 0.99).
    log_dir: string
        Directory where logs will be written (Default is `runs`).
    '''

    def __init__(self, env, gamma=0.99, log_dir='runs'):
        self.env = env
        self.logger = U.Logger(log_dir)
        self.gamma = gamma
        self.rewards = [0]

        self.models = U.SimpleMemory()
        self.last_logged_ep = self.env.num_episodes

    def _check_termination(self):
        if (self.models.policy.num_updates // self.max_updates >= 1
                or self.env.num_episodes // self.max_episodes >= 1
                or self.env.num_steps // self.max_steps >= 1):
            return True

        return False

    def _register_model(self, name, model):
        setattr(self.models, name, model)
        model.attach_logger(self.logger)

    @abstractmethod
    def step(self):
        '''
        This method should be overwritten by a subclass.

        This method is called at each interaction of the training loop,
        and should define the training procedure.
        '''
        pass

    def train(self, max_updates=-1, max_episodes=-1, max_steps=-1):
        '''
        It should define the training loop of the algorithm.

        Parameters
        ----------
        max_updates: int
            Maximum number of gradient updates (Default is -1, meaning it doesn't matter).
        max_episodes: int
            Maximum number of episodes (Default is -1, meaning it doesn't matter).
        max_steps: int
            Maximum number of steps (Default is -1, meaning it doesn't matter).
        '''
        self.max_updates = max_updates
        self.max_episodes = max_episodes
        self.max_steps = max_steps

        while True:
            self.step()
            self.write_logs()

            if self._check_termination():
                break

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
        # TODO: One env needs additional dimension on first (batch) axis
        # return self.models.policy.select_action(state[None])
        return self.models.policy.select_action(state)

    def run_one_episode(self):
        '''
        Run an entire episode using the current model.

        Returns
        -------
        batch: dict
            Dictionary containing information about the episode.
        '''
        return self.env.run_one_episode(select_action_fn=self.select_action)

    def write_logs(self):
        new_eps = abs(self.last_logged_ep - self.env.num_episodes)
        if new_eps != 0:
            self.rewards = self.env.rewards[-new_eps:]
        self.last_logged_ep = self.env.num_episodes

        self.logger.add_log('Reward/Episode', np.mean(self.rewards))

        self.logger.timeit(self.env.num_steps, max_steps=self.max_steps)
        self.logger.log('Update {} | Episode {} | Step {}'.format(
            self.models.policy.num_updates, self.env.num_episodes, self.env.num_steps))

    @classmethod
    def from_config(cls, config, env=None):
        '''
        Create an agent from a configuration object.

        Returns
        -------
        torchrl.agents
            A TorchRL agent.
        '''
        if env is None:
            try:
                env = U.get_obj(config.env.obj)
            except AttributeError:
                raise ValueError('The env must be defined in the config '
                                 'or passed as an argument')

        model = cls._model.from_config(config.model, env.state_info, env.action_info)

        return cls(env, model, **config.agent.as_dict())

    @classmethod
    def from_file(cls, file_path, env=None):
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
        config = U.Config.load(file_path)

        return cls.from_config(config, env=env)
