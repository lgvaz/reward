import gym
from torchrl.envs.base_env import BaseEnv


class GymEnv(BaseEnv):
    '''
    Creates and wraps a gym environment.

    Parameters
    ----------
    env_name: str
        The Gym ID of the env. For a list of available envs check
        `this <https://gym.openai.com/envs/>`_ page.
    wrappers: list
        List of wrappers to be applied on the env.
        Each wrapper should be a function that receives and returns the env.
    '''

    def __init__(self, env_name, wrappers=[], **kwargs):
        self._env_name = env_name
        self.wrappers = wrappers
        self.env = gym.make(env_name)
        # self.env._max_timesteps = 5000
        for wrapper in self.wrappers:
            self.env = wrapper(self.env)
        super().__init__(**kwargs)

    def _reset(self):
        '''
        Calls the reset method on the gym environment.

        Returns
        -------
        state: numpy.ndarray
            A numpy array with the state information.
        '''
        return self.env.reset()

    def _step(self, action):
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
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done

    @property
    def state_info(self):
        '''
        Dictionary containing the shape and type of the state space.
        If it is continuous, also contains the minimum and maximum value.
        '''
        return GymEnv.get_space_info(self.env.observation_space)

    @property
    def action_info(self):
        '''
        Dictionary containing the shape and type of the action space.
        If it is continuous, also contains the minimum and maximum value.
        '''
        return GymEnv.get_space_info(self.env.action_space)

    @property
    def simulator(self):
        return GymEnv

    @property
    def env_name(self):
        return self._env_name

    def update_config(self, config):
        super().update_config(config)
        config.env.obj.update(dict(wrappers=self.wrappers))

    @staticmethod
    def get_space_info(space):
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
            return dict(
                shape=space.shape,
                low_bound=space.low,
                high_bound=space.high,
                dtype='continuous')
        if isinstance(space, gym.spaces.Discrete):
            return dict(shape=space.n, dtype='discrete')
        if isinstance(space, gym.spaces.MultiDiscrete):
            return dict(shape=space.shape, dtype='multi_discrete')
