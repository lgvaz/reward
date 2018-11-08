from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """
    Abstract base class used for implementing new environments.

    Includes some basic functionalities, like the option to use a running mean
    and standard deviation for normalizing states.

    Parameters
    ----------
    env_name: str
        The environment name.
    fixed_normalize_states: bool
        If True, use the state min and max value to normalize the states (Default is False).
    running_normalize_states: bool
        If True, use the running mean and std to normalize the states (Default is False).
    scale_reward: bool
        If True, use the running std to scale the rewards (Default is False).
    """

    def __init__(self, env_name):
        # TODO: Not every env has a env_name, shouldn't be in base
        self.env_name = env_name
        self.env = self._create_env()

    def __str__(self):
        return "<{}>".format(type(self).__name__)

    @property
    @abstractmethod
    def state_space(self):
        """
        Returns a `space` object containing information about the state space.

        Example
        -------
        State space containing 4 continuous observations:

            `return reward.utils.space.Continuous(low=0, high=1, shape=(4,))`
        """

    @property
    @abstractmethod
    def action_space(self):
        """
        Returns a `space` object containing information about the action space.

        Example
        -------
        State space containing 4 continuous actions:

            `return reward.utils.space.Continuous(low=0, high=1, shape=(4,))`
        """

    @abstractmethod
    def _create_env(self):
        """
        Creates ans returns an environment.

        Returns
        -------
            Environment object.
        """

    @abstractmethod
    def reset(self):
        """
        Resets the environment to an initial state.

        Returns
        -------
        numpy.ndarray
            A numpy array with the state information.
        """

    @abstractmethod
    def step(self, action):
        """
        Receives an action and execute it on the environment.

        Parameters
        ----------
        action: int or float or numpy.ndarray
            The action to be executed in the environment, it should be an ``int``
            for discrete enviroments and ``float`` for continuous. There's also
            the possibility of executing multiple actions (if the environment
            supports so), in this case it should be a ``numpy.ndarray``.

        Returns
        -------
        next_state: numpy.ndarray
            A numpy array with the state information.
        reward: float
            The reward.
        done: bool
            Flag indicating the termination of the episode.
        info: dict
            Dict containing additional information about the state.
        """

    @abstractmethod
    def sample_random_action(self):
        pass

    @property
    def num_lives(self):
        raise NotImplementedError

    @property
    def unwrapped(self):
        return self

    def record(self, path):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
