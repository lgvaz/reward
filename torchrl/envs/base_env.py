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
        self.env_name = env_name
        self.env = self._create_env()

    def __str__(self):
        return "<{}>".format(type(self).__name__)

    @abstractmethod
    def get_state_info(self):
        """
        Returns a dict containing information about the state space.

        The dict should contain two keys: ``shape`` indicating the state shape,
        and ``dtype`` indicating the state type.

        Example
        -------
        State space containing 4 continuous actions::

            return dict(shape=(4,), dtype='continuous')
        """

    @abstractmethod
    def get_action_info(self):
        """
        Returns a dict containing information about the action space.

        The dict should contain two keys: ``shape`` indicating the action shape,
        and ``dtype`` indicating the action type.

        If dtype is ``int`` it will be assumed a discrete action space.

        Example
        -------
        Action space containing 4 float numbers::

            return dict(shape=(4,), dtype='float')
        """

    @property
    @abstractmethod
    def simulator(self):
        """
        Returns the name of the simulator being used as a string.
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

    @property
    def num_lives(self):
        raise NotImplementedError

    @property
    def unwrapped(self):
        return self

    def sample_random_action(self):
        raise NotImplementedError

    def record(self, path):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def update_config(self, config):
        """
        Updates a Config object to include information about the environment.

        Parameters
        ----------
        config: Config
            Object used for storing configuration.
        """
        config.new_section(
            "env",
            obj=dict(func=self.simulator, env_name=self.env_name),
            state_info=dict(
                (key, value)
                for key, value in self.get_state_info().items()
                if key not in ("low_bound", "high_bound")
            ),
            action_info=self.get_action_info(),
        )
