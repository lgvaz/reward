from abc import ABC, abstractmethod, abstractproperty


class BaseEnv(ABC):
    '''Base class used for implementing new environments'''

    def __init__(self, config=None):
        pass

    @abstractmethod
    def reset(self):
        '''
        This method should be overwritten by a subclass.

        It should reset the environment to an initial state.

        Returns
        -------
        numpy.ndarray
            A numpy array with the state information.
        '''
        pass

    @abstractmethod
    def step(self, action):
        '''
        This method should be overwritten by a subclass.

        It should receive an action an execute it on the environment.

        Parameters
        ----------
        action: int or float or numpy.ndarray
            The action to be executed in the environment, it should be an ``int`` for
            discrete enviroments and ``float`` for continuous. There's also the possibility
            of executing multiple actions (if the environment supports so),
            in this case it should be a ``numpy.ndarray``.

        Returns
        -------
        next_state: numpy.ndarray
            A numpy array with the state information.
        reward: float
            The reward.
        done: bool
            Flag indicating the termination of the episode.
        '''
        pass

    @abstractproperty
    def state_shape(self):
        '''
        This method should be overwritten by a subclass.

        Should return a dict containing information about the state space.

        The dict should contain two keys: ``shape`` indicating the state shape,
        and ``dtype`` indicating the state type.

        Example
        -------
        State space containing 4 float numbers::

            return dict(shape=(4,), dtype='float')
        '''
        pass

    @abstractproperty
    def action_shape(self):
        '''
        This method should be overwritten by a subclass.

        Should return a dict containing information about the action space.

        The dict should contain two keys: ``shape`` indicating the action shape,
        and ``dtype`` indicating the action type.

        If dtype is ``int`` it will be assumed a discrete action space.

        Example
        -------
        Action space containing 4 float numbers::

            return dict(shape=(4,), dtype='float')
        '''
        pass
