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

    def run(self):
        super().run()
        self._run()

    def _run(self):
        while True:
            data = self.conn.recv()
            #             if data is None:
            #                 break
            if isinstance(data, str) and data == 'reset':
                self.conn.send([env.reset() for env in self.envs])
            else:
                next_states, rewards, dones = [], [], []
                for a, env in zip(data, self.envs):
                    next_state, reward, done = env.step(a)
                    if done:
                        next_state = env.reset()
                        self.shared_rewards.append(env.rewards[-1])

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

        self._create_workers(envs)
        self._states = self.reset()
        self.info_env = envs[0]

    # TODO: DANGEROUS, can call wrong method
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        else:
            return getattr(self.info_env, attr)

    def _create_workers(self, envs):
        WorkerNTuple = namedtuple('Worker', ['process', 'connection'])
        self.workers = []
        for envs in self.split(envs):
            parent_conn, child_conn = Pipe()
            process = EnvWorker(envs, child_conn, self.rewards)
            process.daemon = True
            process.start()
            self.workers.append(WorkerNTuple(process=process, connection=parent_conn))

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

    def record(self, path):
        return self.info_env.record(path)

    def reset(self):
        # Send signal to reset
        for worker in self.workers:
            worker.connection.send('reset')
        # Receive results
        states = []
        for worker in self.workers:
            states.extend(worker.connection.recv())
        return states

    def step(self, actions):
        # Send actions to worker
        for acts, worker in zip(self.split(actions), self.workers):
            worker.connection.send(acts)
        # Receive results
        next_states, rewards, dones = [], [], []
        for worker in self.workers:
            next_state, reward, done = worker.connection.recv()
            next_states.extend(next_state)
            rewards.extend(reward)
            dones.extend(done)

        self.num_steps += self.num_envs

        return next_states, rewards, dones

    def run_one_step(self, select_action_fn):
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
        transitions = []

        for _ in range(num_steps // self.num_envs):
            transition = self.run_one_step(select_action_fn)
            transitions.append(transition)

        return [U.join_transitions(t) for t in zip(*transitions)]

    def split(self, array):
        q, r = divmod(self.num_envs, self.num_workers)
        return [
            array[i * q + min(i, r):(i + 1) * q + min(i + 1, r)]
            for i in range(self.num_workers)
        ]

    def update_config(self, config):
        return self.info_env.update_config(config)
