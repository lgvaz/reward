from abc import ABC, abstractmethod

import numpy as np

import torchrl.utils as U
from torchrl.envs.wrappers import FinalWrapper, StatsRecorder
from torchrl.utils.preprocessing import Normalizer


def profile(x):
    return lambda *args, **kwargs: x(*args, **kwargs)


class BaseEnv(ABC):
    '''
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
    '''

    def __new__(cls, env_name):
        self = super().__new__(cls)
        self.__init__(env_name)
        self = FinalWrapper(self)
        self = StatsRecorder(self)

        return self

    def __init__(self, env_name):
        # *,
        # fixed_normalize_states=False,
        # running_normalize_states=False,
        # running_scale_rewards=False,
        # clip_reward_range=None):
        self.env_name = env_name
        # self.fixed_normalize_states = fixed_normalize_states
        # self.running_normalize_states = running_normalize_states
        # self.running_scale_rewards = running_scale_rewards
        # self.clip_reward_range = clip_reward_range

        self.env = self._create_env()

        # if fixed_normalize_states:
        #     error_msg = ('At least one state value bound is inf, fixed normalization'
        #                  'cant be done, look at running normalization instead')
        #     assert not (abs(
        #         self.get_state_info()['low_bound']) == np.inf).any(), error_msg
        #     assert not (abs(
        #         self.get_state_info()['high_bound']) == np.inf).any(), error_msg

        # self.state_normalizer = Normalizer(
        #     self.get_state_info()['shape']) if running_normalize_states else None
        # self.reward_scaler = Normalizer(1) if running_scale_rewards else None

        # self.num_steps = 0
        # self.ep_reward_sum = 0
        # self.rewards = []
        # self.new_rewards = []
        self._state = None
        # self._raw_state = None
        # self.last_logged_ep = self.num_episodes
        self.num_envs = 1

    def __str__(self):
        return '<{}>'.format(type(self).__name__)

    @abstractmethod
    def get_state_info(self):
        '''
        Returns a dict containing information about the state space.

        The dict should contain two keys: ``shape`` indicating the state shape,
        and ``dtype`` indicating the state type.

        Example
        -------
        State space containing 4 continuous actions::

            return dict(shape=(4,), dtype='continuous')
        '''

    @abstractmethod
    def get_action_info(self):
        '''
        Returns a dict containing information about the action space.

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
        Returns the name of the simulator being used as a string.
        '''

    @abstractmethod
    def _create_env(self):
        '''
        Creates ans returns an environment.

        Returns
        -------
            Environment object.
        '''
        pass

    @abstractmethod
    def reset(self):
        '''
        Resets the environment to an initial state.

        Returns
        -------
        numpy.ndarray
            A numpy array with the state information.
        '''

    @abstractmethod
    def step(self, action):
        '''
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
        '''

    # def auto_step(self, action):
    #     '''
    #     Same as :meth:`step`, but resets the env if done is True.
    #     '''
    #     state, reward, done, info = self.step(action)
    #     if done:
    #         state = self.reset()

    #     return state, reward, done, info

    @property
    def num_lives(self):
        raise NotImplementedError

    @property
    def unwrapped(self):
        return self

    # @property
    # def num_episodes(self):
    #     return len(self.rewards)

    @profile
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
        if self.fixed_normalize_states:
            # TODO: Diving by scalar is 3x faster
            low = self.get_state_info()['low_bound']
            high = self.get_state_info()['high_bound']
            # state = (state - low) / high
            state = np.true_divide(state, 255, dtype=np.float32)

        if self.state_normalizer is not None:
            state = self.state_normalizer.normalize(np.array(state))

        return state.squeeze()

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
        self.ep_reward_sum += reward
        if self.reward_scaler is not None:
            reward = self.reward_scaler.scale(np.array(reward))
        if self.clip_reward_range is not None:
            a_min = -self.clip_reward_range
            a_max = self.clip_reward_range
            reward = np.clip(reward, a_min=a_min, a_max=a_max)

        return reward.squeeze()

    def sample_random_action(self):
        raise NotImplementedError

    def update_normalizers(self):
        '''
        Update mean and var of the normalizers.
        '''
        if self.state_normalizer is not None:
            self.state_normalizer.update()
        # if self.reward_scaler is not None and self.num_episodes > 1:
        # TODO: testing
        if self.reward_scaler is not None:
            self.reward_scaler.update()

    # def reset(self):
    #     '''
    #     Calls the :meth:`_reset` method that should be implemented by a subclass.
    #     The :meth:`_preprocess_state` function is called at the returned state.

    #     Returns
    #     -------
    #     state: numpy.ndarray
    #         The state received by resetting the environment.
    #     '''
    #     raw_state = self._reset()
    #     # TODO: Atari frame already came with the first dim
    #     state = self._preprocess_state(raw_state[None])

    #     return state

    # def step(self, action):
    #     '''
    #     Calls the :meth:`_step` method that should be implemented by a subclass.
    #     The :meth:`_preprocess_state` function is called at the returned state.
    #     The :meth:`_preprocess_reward` function is called at the returned reward.

    #     Parameters
    #     ----------
    #     action: int or float or numpy.ndarray
    #         The action to be executed in the environment, it should be an ``int``
    #         for discrete enviroments and ``float`` for continuous. There's also
    #         the possibility of executing multiple actions (if the environment
    #         supports so), in this case it should be a ``numpy.ndarray``.

    #     Returns
    #     -------
    #     state: numpy.ndarray
    #         The state received by taking the action.
    #     reward: float
    #         The reward received by taking the action.
    #     done: bool
    #         If True the episode is over, and :meth:`reset` should be called.
    #     '''
    #     raw_next_state, reward, done, info = self._step(action)
    #     next_state = self._preprocess_state(np.array(raw_next_state)[None])
    #     reward = self._preprocess_reward(np.array(reward)[None])

    #     # self.num_steps += 1
    #     # if done:
    #     # self.update_normalizers()
    #     # self.rewards.append(self.ep_reward_sum)
    #     # self.ep_reward_sum = 0

    #     return next_state, reward, done, info

    # TODO: WRONG
    # TODO: Name -> auto_step ?
    def run_one_step(self, select_action_fn):
        '''
        Performs a single action on the environment and automatically reset if needed.

        Parameters
        ----------
        select_action_fn: function
            A function that receives the state and returns an action.

        Returns
        -------
        torch.utils.SimpleMemory
            A object containing the transition information.
        '''
        # Choose and execute action
        if self._state is None:
            self._state = self.reset()
        action = select_action_fn(self._state[None]).squeeze()
        next_state, reward, done, info = self.step(action)
        # TODO: INFO AND RAW STATE
        import pdb
        pdb.set_trace()

        transition = U.SimpleMemory(
            # raw_state_t=self._raw_state,
            raw_state_tp1=raw_next_state,
            state_t=self._state,
            state_tp1=next_state,
            action=action,
            reward=reward,
            done=done,
            step=self.num_steps)

        if done:
            self._state = self.reset()
        else:
            self._state = raw_next_state, next_state

        return transition

    def run_one_episode(self, select_action_fn):
        '''
        Performs actions until the end of the episode.

        Parameters
        ----------
        select_action_fn: function
            A function that receives a state and returns an action.

        Returns
        -------
        SimpleMemory
            A ``SimpleMemory`` obj containing information about the trajectory.
        '''
        done = False
        transitions = []

        while not done:
            transition = self.run_one_step(select_action_fn)
            transitions.append(transition)
            done = transition.done

        return [U.join_transitions(transitions)]

    def run_n_episodes(self, select_action_fn, num_episodes):
        '''
        Runs the enviroments for ``num_episodes`` episodes,
        sampling actions from select_action_fn.

        Parameters
        ----------
        select_action_fn: function
            A function that receives a state and returns an action.
        num_episodes: int
            Number of episodes to run.

        Returns
        -------
        SimpleMemory
            A ``SimpleMemory`` obj containing information about the trajectory.
        '''
        return [
            self.run_one_episode(select_action_fn=select_action_fn)[0]
            for _ in range(num_episodes)
        ]

    def run_n_steps(self, select_action_fn, num_steps):
        '''
        Runs the enviroment for ``num_steps`` steps,
        sampling actions from select_action_fn.

        Parameters
        ----------
        select_action_fn: function
            A function that receives a state and returns an action.
        num_steps: int
            Number of steps to run.

        Returns
        -------
        SimpleMemory
            A ``SimpleMemory`` obj containing information about the trajectory.
        '''
        transitions = []

        for _ in range(num_steps):
            transition = self.run_one_step(select_action_fn)
            transitions.append(transition)

        return [U.join_transitions(transitions)]

    def write_logs(self, logger):
        print('Need to write env log')
        # new_eps = abs(self.last_logged_ep - self.num_episodes)
        # if new_eps != 0:
        #     self.new_rewards = self.rewards[-new_eps:]
        # self.last_logged_ep = self.num_episodes

        # logger.add_log('Env/Reward/Episode (New Episodes)', np.mean(self.new_rewards))
        # logger.add_log('Env/Reward/Episode (Last 50)', np.mean(self.rewards[-50:]))

        # if self.state_normalizer is not None:
        #     logger.add_tf_only_log('Env/States/Mean',
        #                            np.mean(self.state_normalizer.means))
        #     logger.add_tf_only_log('Env/States/Vars', np.mean(self.state_normalizer.vars))
        # if self.state_normalizer is not None:
        #     logger.add_tf_only_log('Env/Rewards/Mean', np.mean(self.reward_scaler.means))
        #     logger.add_tf_only_log('Env/Rewards/Vars', np.mean(self.reward_scaler.vars))
        #     # TODO
        #     # logger.add_histogram('Env/Rewards/Vars_hist', self.reward_scaler.vars)

    def record(self, path):
        raise NotImplementedError

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
            state_info=dict((key, value) for key, value in self.get_state_info().items()
                            if key not in ('low_bound', 'high_bound')),
            action_info=self.get_action_info(),
            running_normalize_states=self.running_normalize_states,
            running_scale_rewards=self.running_scale_rewards)
