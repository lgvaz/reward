import gym
from torchrl.envs.base_env import BaseEnv
import torchrl.utils as U


class GymEnv(BaseEnv):
    """
    Creates and wraps a gym environment.

    Parameters
    ----------
    env_name: str
        The Gym ID of the env. For a list of available envs check
        `this <https://gym.openai.com/envs/>`_ page.
    wrappers: list
        List of wrappers to be applied on the env.
        Each wrapper should be a function that receives and returns the env.
    """

    def __init__(self, env_name, **kwargs):

        super().__init__(env_name, **kwargs)

    def _create_env(self):
        env = gym.make(self.env_name)

        return env

    @property
    def simulator(self):
        return GymEnv

    def reset(self):
        """
        Calls the reset method on the gym environment.

        Returns
        -------
        state: numpy.ndarray
            A numpy array with the state information.
        """
        return self.env.reset()

    def step(self, action):
        """
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
        """
        if self.get_action_info().space == "discrete":
            action = int(action)
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

    def record(self, path):
        self.env = Monitor(env=self.env, directory=path, video_callable=lambda x: True)

    def get_state_info(self):
        """
        Dictionary containing the shape and type of the state space.
        If it is continuous, also contains the minimum and maximum value.
        """
        return GymEnv.get_space_info(self.env.observation_space)

    def get_action_info(self):
        """
        Dictionary containing the shape and type of the action space.
        If it is continuous, also contains the minimum and maximum value.
        """
        return GymEnv.get_space_info(self.env.action_space)

    def sample_random_action(self):
        return self.env.action_space.sample()

    def seed(self, value):
        self.env.seed(value)

    def update_config(self, config):
        """
        Updates a Config object to include information about the environment.

        Parameters
        ----------
        config: Config
            Object used for storing configuration.
        """
        super().update_config(config)
        config.env.obj.update(dict(wrappers=self.wrappers))

    def close(self):
        self.env.close()

    @staticmethod
    def get_space_info(space):
        """
        Gets the shape of the possible types of states in gym.

        Parameters
        ----------
        space: gym.spaces
            Space object that describes the valid actions and observations

        Returns
        -------
        dict
            Dictionary containing the space shape and type
        """
        if isinstance(space, gym.spaces.Box):
            return U.memories.SimpleMemory(
                shape=space.shape,
                low_bound=space.low,
                high_bound=space.high,
                space="continuous",
                dtype=space.dtype,
            )
        if isinstance(space, gym.spaces.Discrete):
            return U.memories.SimpleMemory(
                shape=space.n, space="discrete", dtype=space.dtype
            )
        if isinstance(space, gym.spaces.MultiDiscrete):
            return U.memories.SimpleMemory(
                shape=space.shape, space="multi_discrete", dtype=space.dtype
            )
