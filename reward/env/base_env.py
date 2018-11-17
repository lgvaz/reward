from abc import ABC, abstractmethod
from boltons.cacheutils import cachedproperty


class BaseEnv(ABC):
    """
    Abstract base class used for implementing new environments.
    """

    def __init__(self):
        self.env = self._create_env()

    def __str__(self):
        return "<{}>".format(type(self).__name__)

    # TODO: Follow https://github.com/mahmoud/boltons/pull/184
    @cachedproperty
    @abstractmethod
    def s_space(self):
        """
        Returns a `space` object containing information about the state space.

        Example
        -------
        State space containing 4 continuous observations:

            `return reward.utils.space.Continuous(low=0, high=1, shape=(4,))`
        """

    @cachedproperty
    @abstractmethod
    def ac_space(self):
        """
        Returns a `space` object containing information about the action space.

        Example
        -------
        State space containing 4 continuous acs:

            `return reward.utils.space.Continuous(low=0, high=1, shape=(4,))`
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
    def step(self, ac):
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
    def _create_env(self):
        """
        Creates ans returns an environment.

        Returns
        -------
            Environment object.
        """

    @property
    def num_lives(self): raise NotImplementedError

    @property
    def unwrapped(self): return self

    def sample_random_ac(self): return self.ac_space.sample()

    def record(self, path): raise NotImplementedError

    def close(self): raise NotImplementedError
