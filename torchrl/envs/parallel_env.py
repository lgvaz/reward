import numpy as np
import multiprocessing
from collections import namedtuple
from multiprocessing import Manager, Pipe, Process

import torchrl.utils as U
import numpy as np
from ctypes import c_uint, c_float, c_double, c_int
from multiprocessing.sharedctypes import RawArray


class EnvWorker(Process):
    def __init__(self, envs, conn, shared_transition):
        super().__init__()
        self.envs = envs
        self.conn = conn
        self.shared_tran = shared_transition

    def run(self):
        super().run()
        self._run()

    def _run(self):
        while True:
            data = self.conn.recv()

            if data is None:
                for i, env in enumerate(self.envs):
                    self.shared_tran.state[i] = env._reset()

            else:
                for i, (a, env) in enumerate(zip(self.shared_tran.action, self.envs)):
                    next_state, reward, done = env._step(a)

                    if done:
                        next_state = env._reset()

                    self.shared_tran.state[i] = next_state
                    self.shared_tran.reward[i] = reward
                    self.shared_tran.done[i] = done

            self.conn.send(True)


class ParallelEnv:
    r'''
    The parallelization is done as described in
    [this paper](https://arxiv.org/pdf/1705.04862.pdf).
    Heavily inspired in code from [here](https://github.com/Alfredvc/paac).

    Each worker will hold :math:`\frac{num\_envs}{num\_workers}` envs.

    Parameters
    ----------
    envs: list
        A list of all the torchrl envs.
    num_workers: int
        How many process to spawn (Default is available number of CPU cores).
    '''
    NUMPY_TO_C_DTYPE = {
        np.float32: c_float,
        np.float64: c_double,
        np.uint8: c_uint,
        np.int32: c_int
    }

    def __init__(self, envs, num_workers=None):
        self.num_envs = len(envs)
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.num_steps = 0
        self.manager = Manager()
        self.envs_rewards = np.zeros(self.num_envs)

        assert self.num_envs >= self.num_workers, \
            'Number of envs must be greater or equal the number of workers'

        # Extract information from the envs
        env = envs[0]
        self.state_normalizer = env.state_normalizer
        self.reward_scaler = env.reward_scaler
        self.root_env = env
        self.rewards = self.root_env.rewards

        self._create_shared_transitions(envs)
        self._create_workers(envs)
        self._states = None
        self._raw_states = None

    @property
    def state_info(self):
        return self.root_env.state_info

    @property
    def action_info(self):
        return self.root_env.action_info

    @property
    def simulator(self):
        return self.root_env.simulator

    @property
    def num_episodes(self):
        return self.root_env.num_episodes

    def _create_shared_transitions(self, envs):
        # TODO: dtype for atari
        state = self._get_shared(
            np.zeros([self.num_envs] + list(self.state_info['shape']), dtype=np.float32))
        reward = self._get_shared(np.zeros(self.num_envs, dtype=np.float32))
        done = self._get_shared(np.zeros(self.num_envs, dtype=np.float32))

        action_info = self.root_env.action_info
        if action_info['dtype'] == 'continuous':
            shape = (self.num_envs, np.prod(action_info['shape']))
            action_type = np.float32
        elif action_info['dtype'] == 'discrete':
            shape = (self.num_envs, )
            action_type = np.int32
        else:
            raise ValueError('Action dtype {} not implemented'.format(action_info.dtype))
        action = self._get_shared(np.zeros(shape, dtype=action_type))

        self.shared_tran = U.SimpleMemory(
            state=state, reward=reward, done=done, action=action)

    def _create_workers(self, envs):
        '''
        Creates and starts each worker in a distinct process.

        Parameters
        ----------
        envs: list
            List of envs, each worker will have approximately the same number of envs.
        '''
        WorkerNTuple = namedtuple('Worker', ['process', 'connection'])
        self.workers = []

        for envs_i, s_s, s_r, s_d, s_a in zip(
                self.split(envs),
                self.split(self.shared_tran.state),
                self.split(self.shared_tran.reward),
                self.split(self.shared_tran.done), self.split(self.shared_tran.action)):

            shared_tran = U.SimpleMemory(state=s_s, reward=s_r, done=s_d, action=s_a)
            parent_conn, child_conn = Pipe()

            process = EnvWorker(
                envs=envs_i, conn=child_conn, shared_transition=shared_tran)
            process.daemon = True
            process.start()

            self.workers.append(WorkerNTuple(process=process, connection=parent_conn))

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
        return self.root_env._preprocess_state(state)

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
        return self.root_env._preprocess_reward(reward)

    def sync(self):
        for worker in self.workers:
            worker.connection.recv()

    def update_normalizers(self):
        '''
        Update mean and var of the normalizers.
        '''
        self.root_env.update_normalizers()

    def record(self, path):
        return self.root_env.record(path)

    def reset(self):
        '''
        Reset all workers in parallel, using Pipe for communication.
        '''
        # Send signal to reset
        for worker in self.workers:
            worker.connection.send(None)
        # Receive results
        raw_states = self.shared_tran.state
        self.sync()

        states = self._preprocess_state(raw_states)
        return raw_states, states

    def step(self, actions):
        '''
        Step all workers in parallel, using Pipe for communication.

        Parameters
        ----------
        action: int or float or numpy.ndarray
            The action to be executed in the environment, it should be an ``int``
            for discrete enviroments and ``float`` for continuous. There's also
            the possibility of executing multiple actions (if the environment
            supports so), in this case it should be a ``numpy.ndarray``.
        '''
        # Send actions to worker
        self.shared_tran.action[...] = actions
        for worker in self.workers:
            worker.connection.send(True)
        self.sync()
        self.num_steps += self.num_envs

        raw_next_states = self.shared_tran.state
        rewards = self.shared_tran.reward
        dones = self.shared_tran.done

        # Accumulate rewards
        self.envs_rewards += rewards
        for i, done in enumerate(dones):
            if done:
                self.rewards.append(self.envs_rewards[i])
                self.envs_rewards[i] = 0

        next_states = self._preprocess_state(raw_next_states)
        rewards = self._preprocess_reward(rewards)
        self.update_normalizers()

        return raw_next_states, next_states, rewards, dones

    def run_one_step(self, select_action_fn):
        '''
        Performs a single action on each environment and automatically reset if needed.

        Parameters
        ----------
        select_action_fn: function
            A function that receives the states and returns the actions.

        Returns
        -------
        torch.utils.SimpleMemory
            A object containing the transition information.
        '''
        if self._states is None:
            self._raw_states, self._states = self.reset()

        actions = select_action_fn(np.array(self._states))
        raw_next_states, next_states, rewards, dones = self.step(actions)

        transition = [
            U.SimpleMemory(
                raw_state_t=rst,
                raw_state_tp1=rstp1,
                state_t=st,
                state_tp1=stp1,
                action=act,
                reward=rew,
                step=self.num_steps,
                done=d)
            for rst, rstp1, st, stp1, act, rew, d in
            zip(self._raw_states, raw_next_states, self._states, next_states, actions,
                rewards, dones)
        ]

        self._raw_states = raw_next_states
        self._states = next_states

        return transition

    def run_n_steps(self, select_action_fn, num_steps):
        '''
        Runs the enviroments for ``num_steps`` steps,
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

        for _ in range(num_steps // self.num_envs):
            transition = self.run_one_step(select_action_fn)
            transitions.append(transition)

        return [U.join_transitions(t) for t in zip(*transitions)]

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
        transitions = []
        dones = 0

        while dones < num_episodes:
            transition = self.run_one_step(select_action_fn)
            transitions.append(transition)

            dones += sum(t['done'] for t in transition)

        return [U.join_transitions(t) for t in zip(*transitions)]

    def split(self, array):
        '''
        Divide the input in approximately equal chunks for all workers.

        Parameters
        ----------
        array: array or list
            The object to be divided.

        Returns
        -------
        list
            The divided object.
        '''
        q, r = divmod(self.num_envs, self.num_workers)
        return [
            array[i * q + min(i, r):(i + 1) * q + min(i + 1, r)]
            for i in range(self.num_workers)
        ]

    def write_logs(self, logger):
        return self.root_env.write_logs(logger=logger)

    def update_config(self, config):
        return self.root_env.update_config(config)

    def _get_shared(self, array):
        """
        Returns a RawArray backed numpy array that can be shared between processes.
        :param array: the array to be shared
        :return: the RawArray backed numpy array
        """

        dtype = self.NUMPY_TO_C_DTYPE[array.dtype.type]

        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        return np.frombuffer(shared, dtype).reshape(shape)
