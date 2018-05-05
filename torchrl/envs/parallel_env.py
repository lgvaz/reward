import numpy as np
import multiprocessing
from collections import namedtuple
from multiprocessing import Pipe, Process, Manager

import torchrl.utils as U


class EnvWorker(Process):
    def __init__(self, envs, conn, shared_rewards):
        super().__init__()
        self.envs = envs
        self.conn = conn
        self.shared_rewards = shared_rewards
        self.reward_sum = 0

    def run(self):
        super().run()
        self._run()

    def _run(self):
        while True:
            data = self.conn.recv()

            if isinstance(data, str) and data == 'reset':
                self.conn.send([env.reset() for env in self.envs])

            else:
                next_states, rewards, dones = [], [], []
                for a, env in zip(data, self.envs):
                    next_state, reward, done = env._step(a)
                    self.reward_sum += reward

                    if done:
                        next_state = env._reset()
                        self.shared_rewards.append(self.reward_sum)
                        self.reward_sum = 0

                    next_states.append(next_state)
                    rewards.append(reward)
                    dones.append(done)

                self.conn.send([next_states, rewards, dones])


class ParallelEnv:
    r'''
    The parallelization is done as described in
    [this paper](https://arxiv.org/pdf/1705.04862.pdf).

    Each worker will hold :math:`\frac{num_envs}{num_workers}` envs.

    Parameters
    ----------
    envs: list
        A list of all the torchrl envs.
    num_workers: int
        How many process to spawn (Default is available number of CPU cores).
    '''

    def __init__(self, envs, num_workers=None):
        self.num_envs = len(envs)
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.num_steps = 0
        self.manager = Manager()
        self.rewards = self.manager.list()

        assert self.num_envs >= self.num_workers, \
            'Number of envs must be greater or equal the number of workers'

        # Extract information from the envs
        env = envs[0]
        self.state_normalizer = env.state_normalizer
        self.reward_scaler = env.reward_scaler
        self.info_env = env

        self._create_workers(envs)
        self._states = None

    @property
    def state_info(self):
        return self.info_env.state_info

    @property
    def action_info(self):
        return self.info_env.action_info

    @property
    def simulator(self):
        return self.info_env.simulator

    @property
    def num_episodes(self):
        return len(self.rewards)

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

        for envs_i in self.split(envs):
            parent_conn, child_conn = Pipe()
            process = EnvWorker(envs_i, child_conn, self.rewards)
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
        if self.state_normalizer is not None:
            state = self.state_normalizer.normalize(np.array(state))

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
        # TODO: Atari rewards are clipped, we want the unclipped rewards
        if self.reward_scaler is not None:
            reward = self.reward_scaler.scale(np.array(reward)).squeeze()
        return reward

    def update_normalizers(self):
        '''
        Update mean and var of the normalizers.
        '''
        if self.state_normalizer is not None:
            self.state_normalizer.update()
        if self.reward_scaler is not None and self.num_episodes > 1:
            self.reward_scaler.update()

    def record(self, path):
        return self.info_env.record(path)

    def reset(self):
        '''
        Reset all workers in parallel, using Pipe for communication.
        '''
        # Send signal to reset
        for worker in self.workers:
            worker.connection.send('reset')
        # Receive results
        states = np.concatenate([worker.connection.recv() for worker in self.workers])

        states = self._preprocess_state(states)
        return states

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
        for acts, worker in zip(self.split(actions), self.workers):
            worker.connection.send(acts)
        # Receive results
        next_states, rewards, dones = map(
            np.concatenate, zip(*[worker.connection.recv() for worker in self.workers]))
        self.num_steps += self.num_envs

        next_states = self._preprocess_state(next_states)
        rewards = self._preprocess_reward(rewards)
        self.update_normalizers()

        return next_states, rewards, dones

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
            self._states = self.reset()

        actions = select_action_fn(np.array(self._states))
        next_states, rewards, dones = self.step(actions)

        transition = [
            U.SimpleMemory(state_t=st, state_tp1=stp1, action=act, reward=rew, done=d)
            for st, stp1, act, rew, d in zip(self._states, next_states, actions, rewards,
                                             dones)
        ]

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

    def update_config(self, config):
        return self.info_env.update_config(config)
