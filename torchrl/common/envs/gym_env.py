import gym
from torchrl.common.envs.base_env import BaseEnv


class GymEnv(BaseEnv):
    '''Creates and wraps a gym environment'''

    def __init__(self, env_name, wrappers=None):
        self.env = gym.make(env_name)
        if wrappers is not None:
            self.env = wrappers(self.env)
        pass

    def reset(self):
        '''
        Calls the reset method on the gym environment.

        Returns
        -------
        state: numpy.ndarray
            A numpy array with the state information.
        '''
        return self.env.reset()

    def step(self, state):
        '''
        Calls the step method on the gym environment.

        Parameters
        ----------
        action: int or float or numpy.ndarray
            The action to be executed in the environment, it should be an int for
            discrete enviroments and float for continuous. There's also the possibility
            of executing multiple actions (if the environment supports so),
            in this case it should be a numpy.ndarray.

        Returns
        -------
        next_state: numpy.ndarray
            A numpy array with the state information.
        reward: float
            The reward.
        done: bool
            Flag indicating the termination of the episode.
        '''
        next_state, reward, done, _ = self.env.step(state)
        return next_state, reward, done

    @property
    def state_shape(self):
        '''Shape of the state space'''
        return GymEnv.get_space_shape(self.env.observation_space)

    @property
    def action_shape(self):
        '''Shape of the action space'''
        return GymEnv.get_space_shape(self.env.action_space)

    @staticmethod
    def get_space_shape(space):
        '''
        Gets the shape of the possible types of states in gym.

        Parameters
        ----------
        space: gym.spaces
            Space object that describes the valid actions and observations

        Returns
        -------
        dict
            Dictionary containing the space shape and type
        '''
        if isinstance(space, gym.spaces.Box):
            return dict(shape=space.shape, dtype='float')
        if isinstance(space, gym.spaces.Discrete):
            return dict(shape=space.n, dtype='int')
        if isinstance(space, gym.spaces.MultiDiscrete):
            return dict(shape=space.shape, dtype='int')
