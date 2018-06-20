import numpy as np
import multiprocessing
from collections import namedtuple
from multiprocessing import Manager, Pipe, Process, Queue

import torchrl.utils as U
import numpy as np
from ctypes import c_uint8, c_float, c_double, c_int
from multiprocessing.sharedctypes import RawArray


def profile(x):
    return lambda *args, **kwargs: x(*args, **kwargs)


class EnvWorker(Process):
    def __init__(self, envs, conn, barrier, shared_transition):
        super().__init__()
        self.envs = envs
        self.conn = conn
        self.barrier = barrier
        self.shared_tran = shared_transition

    def run(self):
        super().run()
        self._run()

    def _run(self):
        while True:
            data = self.conn.get()

            if data is None:
                for i, env in enumerate(self.envs):
                    self.shared_tran.state[i] = env.reset()

            else:
                for i, (a, env) in enumerate(zip(self.shared_tran.action, self.envs)):
                    next_state, reward, done, info = env.step(a, auto_reset=True)

                    self.shared_tran.state[i] = next_state
                    self.shared_tran.reward[i] = reward
                    self.shared_tran.done[i] = done
                    self.shared_tran.info[i].update(info)

            # self.conn.put(True)
            self.barrier.send(True)


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
        np.uint8: c_uint8,
        np.int32: c_int,
        np.int64: c_int
    }

    def __init__(self, envs, num_workers=None, seed=None):
        self.num_envs = len(envs)
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.num_steps = 0
        self.manager = Manager()
        self.envs_rewards = np.zeros(self.num_envs)

        assert self.num_envs >= self.num_workers, \
            'Number of envs must be greater or equal the number of workers'

        # Extract information from the envs
        env = envs[0]
        # self.state_normalizer = env.state_normalizer
        # self.reward_scaler = env.reward_scaler
        self.root_env = env
        self.rewards = self.root_env.rewards

        self._set_seeds(envs, seed)
        self._create_shared_transitions()
        self._create_workers(envs)
        self._states = None
        self._raw_states = None

        self.batch = None
        self.steps_per_batch = None

    def get_state_info(self):
        info = self.root_env.get_state_info()
        #     info.shape = (self.num_envs, ) + info.shape[1:]
        return info

    def get_action_info(self):
        return self.root_env.get_action_info()

    def sample_random_action(self):
        return np.array(
            [self.root_env.sample_random_action() for _ in range(self.num_envs)])

    @property
    def simulator(self):
        return self.root_env.simulator

    @property
    def num_episodes(self):
        return self.root_env.num_episodes

    def _create_batch(self, num_steps):
        # TODO
        # TODO: dtype for atari
        # TODO
        if self.batch is None or self.steps_per_batch != num_steps:
            self.steps_per_batch = num_steps
            state_t_and_tp1 = np.empty(
                [num_steps + 1, self.num_envs] + list(self.get_state_info().shape),
                dtype=self.root_env.get_state_info().dtype)
            action = self._get_action_array()
            action = np.empty((num_steps, *action.shape), dtype=action.dtype)
            reward = np.empty([num_steps] + [self.num_envs], dtype=np.float32)
            done = np.empty([num_steps] + [self.num_envs])

            self.batch = U.Batch(
                dict(
                    state_t_and_tp1=state_t_and_tp1,
                    action=action,
                    reward=reward,
                    done=done))

        return self.batch

    def _create_shared_transitions(self):
        # TODO
        # TODO: dtype for atari
        # TODO
        state = self._get_shared(
            np.zeros(
                [self.num_envs] + list(self.get_state_info().shape),
                dtype=self.root_env.get_state_info().dtype))
        action = self._get_shared(self._get_action_array())
        reward = self._get_shared(np.zeros(self.num_envs, dtype=np.float32))
        done = self._get_shared(np.zeros(self.num_envs, dtype=np.float32))
        #         info = self.manager.list([dict()] * self.num_envs)
        info = [self.manager.dict() for _ in range(self.num_envs)]

        self.shared_tran = U.SimpleMemory(
            state=state, reward=reward, done=done, action=action, info=info)

    def _get_action_array(self):
        action_info = self.root_env.get_action_info()
        if action_info.space == 'continuous':
            shape = (self.num_envs, np.prod(action_info.shape))
            # action_type = np.float32
        elif action_info.space == 'discrete':
            shape = (self.num_envs, )
            # action_type = np.int32
        else:
            raise ValueError('Action dtype {} not implemented'.format(action_info.dtype))
        return np.zeros(shape, dtype=action_info.dtype)

    def _create_workers(self, envs):
        '''
        Creates and starts each worker in a distinct process.

        Parameters
        ----------
        envs: list
            List of envs, each worker will have approximately the same number of envs.
        '''
        WorkerNTuple = namedtuple('Worker', ['process', 'connection', 'barrier'])
        self.workers = []

        for envs_i, s_s, s_r, s_d, s_a, s_i in zip(
                self.split(envs),
                self.split(self.shared_tran.state),
                self.split(self.shared_tran.reward),
                self.split(self.shared_tran.done),
                self.split(self.shared_tran.action), self.split(self.shared_tran.info)):

            shared_tran = U.SimpleMemory(
                state=s_s, reward=s_r, done=s_d, action=s_a, info=s_i)
            parent_conn, child_conn = Pipe()
            queue = Queue()
            # barrier = Queue()

            # process = EnvWorker(
            #     envs=envs_i, conn=child_conn, shared_transition=shared_tran)
            process = EnvWorker(
                envs=envs_i,
                conn=queue,
                barrier=child_conn,
                shared_transition=shared_tran)
            process.daemon = True
            process.start()

            # self.workers.append(WorkerNTuple(process=process, connection=parent_conn))
            self.workers.append(
                WorkerNTuple(process=process, connection=queue, barrier=parent_conn))

    def _get_shared(self, array):
        """
        A numpy array that can be shared between processes.
        From: `alfredcv <https://sourcegraph.com/github.com/Alfredvc/paac/-/blob/runners.py#L20:9-20:20$references>`_.

        Parameters
        ----------
        array: np.array
            The shared to be shared

        Returns
        -------
        A shared numpy array.
        """

        dtype = self.NUMPY_TO_C_DTYPE[array.dtype.type]

        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        return np.frombuffer(shared, dtype).reshape(shape)

    # def _preprocess_state(self, state):
    #     '''
    #     Perform transformations on the state (scaling, normalizing, cropping, etc).

    #     Parameters
    #     ----------
    #     state: numpy.ndarray
    #         The state to be processed.

    #     Returns
    #     -------
    #     state: numpy.ndarray
    #         The transformed state.
    #     '''
    #     return self.root_env._preprocess_state(state)

    # def _preprocess_reward(self, reward):
    #     '''
    #     Perform transformations on the reward e.g. clipping.

    #     Parameters
    #     ----------
    #     reward: float
    #         The reward to be processed.

    #     Returns
    #     -------
    #     reward: float
    #         The transformed reward.
    #     '''
    #     return self.root_env._preprocess_reward(reward)

    # def update_normalizers(self):
    #     '''
    #     Update mean and var of the normalizers.
    #     '''
    #     self.root_env.update_normalizers()

    def sync(self):
        for worker in self.workers:
            worker.barrier.recv()

    def record(self, path):
        return self.root_env.record(path)

    def reset(self):
        '''
        Reset all workers in parallel, using Pipe for communication.
        '''
        # Send signal to reset
        for worker in self.workers:
            worker.connection.put(None)
        # Receive results
        self.sync()
        states = self.shared_tran.state.copy()

        # states = self._preprocess_state(raw_states)
        return states

    @profile
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
            worker.connection.put(True)
        self.sync()
        self.num_steps += self.num_envs

        next_states = self.shared_tran.state.copy()
        rewards = self.shared_tran.reward.copy()
        dones = self.shared_tran.done.copy()
        infos = self.shared_tran.info.copy()

        # Accumulate rewards
        self.envs_rewards += rewards
        for i, done in enumerate(dones):
            if done:
                self.rewards.append(self.envs_rewards[i])
                self.envs_rewards[i] = 0

        # next_states = self._preprocess_state(raw_next_states)
        # rewards = self._preprocess_reward(rewards)
        # self.update_normalizers()

        return next_states, rewards, dones, infos

    # @profile
    # # TODO: Name -> auto_step ?
    # def run_one_step(self, select_action_fn):
    #     '''
    #     Performs a single action on each environment and automatically reset if needed.

    #     Parameters
    #     ----------
    #     select_action_fn: function
    #         A function that receives the states and returns the actions.

    #     Returns
    #     -------
    #     torch.utils.SimpleMemory
    #         A object containing the transition information.
    #     '''
    #     if self._states is None:
    #         # self._raw_states, self._states = self.reset()
    #         self._states = self.reset()

    #     actions = select_action_fn(np.array(self._states))
    #     next_states, rewards, dones = self.step(actions)

    #     # TODO: raw states
    #     # transition = [
    #     #     U.SimpleMemory(
    #     #         # raw_state_t=rst,
    #     #         # raw_state_tp1=rstp1,
    #     #         state_t=st,
    #     #         state_tp1=stp1,
    #     #         action=act,
    #     #         reward=rew,
    #     #         step=self.num_steps,
    #     #         done=d)
    #     #     for rst, rstp1, st, stp1, act, rew, d in
    #     #     zip(self._raw_states, raw_next_states, self._states, next_states, actions,
    #     #         rewards, dones)
    #     # ]

    #     # self._raw_states = raw_next_states
    #     self._states = next_states

    #     return self._states, next_states, actions, rewards, dones

    # @profile
    # def run_n_steps(self, select_action_fn, num_steps):
    #     '''
    #     Runs the enviroments for ``num_steps`` steps,
    #     sampling actions from select_action_fn.

    #     Parameters
    #     ----------
    #     select_action_fn: function
    #         A function that receives a state and returns an action.
    #     num_steps: int
    #         Number of steps to run.

    #     Returns
    #     -------
    #     SimpleMemory
    #         A ``SimpleMemory`` obj containing information about the trajectory.
    #     '''
    #     horizon = num_steps // self.num_envs
    #     batch = self._create_batch(horizon)
    #     # transitions = []

    #     for i in range(horizon):
    #         state_t, state_tp1, action, reward, done = self.run_one_step(select_action_fn)
    #         # transitions.append(transition)
    #         batch.state_t_and_tp1[i] = state_t
    #         batch.action[i] = action
    #         batch.reward[i] = reward
    #         batch.done[i] = done
    #     batch.state_t_and_tp1[i + 1] = state_tp1

    #     batch.state_t = batch.state_t_and_tp1[:-1]
    #     batch.state_tp1 = batch.state_t_and_tp1[1:]

    #     # return [U.join_transitions(t) for t in zip(*transitions)]
    #     return batch

    # def run_n_episodes(self, select_action_fn, num_episodes):
    #     '''
    #     Runs the enviroments for ``num_episodes`` episodes,
    #     sampling actions from select_action_fn.

    #     Parameters
    #     ----------
    #     select_action_fn: function
    #         A function that receives a state and returns an action.
    #     num_episodes: int
    #         Number of episodes to run.

    #     Returns
    #     -------
    #     SimpleMemory
    #         A ``SimpleMemory`` obj containing information about the trajectory.
    #     '''
    #     transitions = []
    #     dones = 0

    #     while dones < num_episodes:
    #         transition = self.run_one_step(select_action_fn)
    #         transitions.append(transition)

    #         dones += sum(t['done'] for t in transition)

    #     return [U.join_transitions(t) for t in zip(*transitions)]

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

    def _set_seeds(self, envs, seed):
        if seed is not None:
            np.random.seed(seed)
            seeds = np.random.choice(10000, size=self.num_envs)

            for i, (env, s) in enumerate(zip(envs, seeds)):
                print('Seed for env {}: {}'.format(i, s))
                env.seed(int(s))
