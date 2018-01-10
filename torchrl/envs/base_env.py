from abc import ABC, abstractmethod

import numpy as np

from torchrl.utils.preprocessing import Normalizer


class BaseEnv(ABC):
    '''
    Abstract base class used for implementing new environments.

    Includes some basic functionalities, like the option to use a running mean
    and standard deviation for normalizing states.

    Parameters
    ----------
    normalize_states: bool
        If True, normalize the states (Default is True).
    '''

    def __init__(self, normalize_states=False):
        self.normalize_states = normalize_states
        if normalize_states:
            self.normalizer = Normalizer(self.state_info['shape'])
        else:
            self.normalizer = None

        self._state = self.reset()

    @abstractmethod
    def _reset(self):
        '''
        This method should be overwritten by a subclass.

        It should reset the environment to an initial state.

        Returns
        -------
        numpy.ndarray
            A numpy array with the state information.
        '''

    @abstractmethod
    def _step(self, action):
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

    def _preprocess_state(self, state):
        '''
        Perform transformations on the state (scaling, normalizing, cropping, etc).

        Parameters
        ----------
        state: numpy.ndarray
            The state to be processed.

        Returns
        -------
        state: numpy.ndarray
            The transformed state.
        '''
        if self.normalizer is not None:
            state = self.normalizer.normalize(state)

        return state

    def _preprocess_reward(self, reward):
        '''
        Perform transformations on the reward e.g. clipping.

        Parameters
        ----------
        reward: float
            The reward to be processed.

        Returns
        -------
        reward: float
            The transformed reward.
        '''
        return reward

    def reset(self):
        '''
        Calls the reset method that should be implemented by a subclass.
        The :meth:`_preprocess_state` function is called at the returned state.

        Returns
        -------
        state: numpy.ndarray
            The state received by resetting the environment.
        '''
        state = self._reset()
        state = self._preprocess_state(state)

        if self.normalizer is not None:
            self.normalizer.update()

        return state

    def step(self, action):
        next_state, reward, done = self._step(action)
        next_state = self._preprocess_state(next_state)
        reward = self._preprocess_reward(reward)

        return next_state, reward, done

    @property
    @abstractmethod
    def state_info(self):
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

    @property
    @abstractmethod
    def action_info(self):
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

    @property
    @abstractmethod
    def simulator(self):
        '''
        This method should be overwritten by a subclass.

        Should return the name of the simulator being used as a string.
        '''

    @property
    @abstractmethod
    def env_name(self):
        '''
        This method should be overwritten by a subclass.

        Should return the name of the environment.
        '''

    def update_config(self, config):
        '''
        Updates a Config object to include information about the environment.

        Parameters
        ----------
        config: Config
            Object used for storing configuration.
        '''
        config.new_section(
            'env',
            obj=dict(func=self.simulator, env_name=self.env_name),
            state_info=dict((key, value) for key, value in self.state_info.items()
                            if key not in ('low_bound', 'high_bound')),
            action_info=self.action_info,
            normalize_states=self.normalize_states)

    def run_one_step(self, select_action_fn):
        # Choose and execute action
        action = select_action_fn(self._state)
        next_state, reward, done = self.step(action)

        transition = dict(
            state_t=self._state,
            state_tp1=next_state,
            action=action,
            reward=reward,
            done=done)

        if done:
            self._state = self.reset()
        else:
            self._state = next_state

        return transition

    def run_one_episode(self, **kwargs):
        done = False
        transitions = []

        while not done:
            transition = self.run_one_step(**kwargs)
            transitions.append(transition)
            done = transition['done']

        trajectory = {
            key + 's': np.array([t[key] for t in transitions])
            for key in transitions[0]
        }

        return trajectory
