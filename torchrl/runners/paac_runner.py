import multiprocessing
from collections import namedtuple
from ctypes import c_double, c_float, c_int, c_uint8

# from torch.multiprocessing import Manager, Pipe, Process, Queue
from multiprocessing import Manager, Pipe, Process, Queue
from multiprocessing.sharedctypes import RawArray

import numpy as np

import torchrl.utils as U
from torchrl.runners import BaseRunner


class PAACRunner(BaseRunner):
    NUMPY_TO_C_DTYPE = {
        np.float32: c_float,
        np.float64: c_double,
        np.uint8: c_uint8,
        np.int32: c_int,
        np.int64: c_int,
    }

    def __init__(self, env, num_workers=None):
        super().__init__(env=env)
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self._envs_rewards_sum = np.zeros(self.num_envs)
        self._envs_ep_lengths = np.zeros(self.num_envs)
        self.manager = Manager()

        self._create_shared_transitions()
        self._create_workers()

    @property
    def num_envs(self):
        return len(self.env)

    def _create_shared_transitions(self):
        state = self._get_shared(
            np.zeros(self.get_state_info().shape, dtype=self.get_state_info().dtype)
        )
        action = self._get_shared(self._get_action_array())
        reward = self._get_shared(np.zeros(self.num_envs, dtype=np.float32))
        done = self._get_shared(np.zeros(self.num_envs, dtype=np.float32))
        info = [self.manager.dict() for _ in range(self.num_envs)]

        self.shared_tran = U.memories.SimpleMemory(
            state=state, reward=reward, done=done, action=action, info=info
        )

    def _create_workers(self):
        """
        Creates and starts each worker in a distinct process.

        Parameters
        ----------
        envs: list
            List of envs, each worker will have approximately the same number of envs.
        """
        WorkerNTuple = namedtuple("Worker", ["process", "connection", "barrier"])
        self.workers = []

        for envs_i, s_s, s_r, s_d, s_a, s_i in zip(
            self.split(self.env),
            self.split(self.shared_tran.state),
            self.split(self.shared_tran.reward),
            self.split(self.shared_tran.done),
            self.split(self.shared_tran.action),
            self.split(self.shared_tran.info),
        ):

            shared_tran = U.memories.SimpleMemory(
                state=s_s, reward=s_r, done=s_d, action=s_a, info=s_i
            )
            parent_conn, child_conn = Pipe()
            queue = Queue()

            process = EnvWorker(
                envs=envs_i,
                conn=queue,
                barrier=child_conn,
                shared_transition=shared_tran,
            )
            process.daemon = True
            process.start()

            self.workers.append(
                WorkerNTuple(process=process, connection=queue, barrier=parent_conn)
            )

    def _get_action_array(self):
        action_info = self.get_action_info()
        if action_info.space == "continuous":
            shape = (self.num_envs, np.prod(action_info.shape))
        elif action_info.space == "discrete":
            shape = (self.num_envs,)
        else:
            raise ValueError(
                "Action dtype {} not implemented".format(action_info.dtype)
            )

        return np.zeros(shape, dtype=action_info.dtype)

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

    def act(self, action):
        # Send actions to worker
        self.shared_tran.action[...] = action
        for worker in self.workers:
            worker.connection.put(True)
        self.sync()
        self.num_steps += self.num_envs

        next_states = self.shared_tran.state.copy()
        rewards = self.shared_tran.reward.copy()
        dones = self.shared_tran.done.copy()
        infos = list(map(dict, self.shared_tran.info))

        # Accumulate rewards
        self._envs_rewards_sum += rewards
        self._envs_ep_lengths += 1
        for i, done in enumerate(dones):
            if done:
                self.rewards.append(self._envs_rewards_sum[i])
                self.ep_lengths.append(self._envs_ep_lengths[i])
                self._envs_rewards_sum[i] = 0
                self._envs_ep_lengths[i] = 0

        return next_states, rewards, dones, infos

    def reset(self):
        """
        Reset all workers in parallel, using Pipe for communication.
        """
        # Send signal to reset
        for worker in self.workers:
            worker.connection.put(None)
        # Receive results
        self.sync()
        states = self.shared_tran.state.copy()

        return states

    def sample_random_action(self):
        return np.array([env.sample_random_action() for env in self.env])

    def sync(self):
        for worker in self.workers:
            worker.barrier.recv()

    def split(self, array):
        """
        Divide the input in approximately equal chunks for all workers.

        Parameters
        ----------
        array: array or list
            The object to be divided.

        Returns
        -------
        list
            The divided object.
        """
        q, r = divmod(self.num_envs, self.num_workers)
        return [
            array[i * q + min(i, r) : (i + 1) * q + min(i + 1, r)]
            for i in range(self.num_workers)
        ]

    def get_state_info(self):
        info = self.env[0].get_state_info()
        info.shape = (self.num_envs,) + info.shape
        return info

    def get_action_info(self):
        return self.env[0].get_action_info()

    def terminate_workers(self):
        for worker in self.workers:
            worker.process.terminate()

    def close(self):
        self.terminate_workers()
        for env in self.env:
            env.close()


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
                    next_state, reward, done, info = env.step(a)

                    if done:
                        next_state = env.reset()

                    self.shared_tran.state[i] = next_state
                    self.shared_tran.reward[i] = reward
                    self.shared_tran.done[i] = done
                    self.shared_tran.info[i].update(info)

            self.barrier.send(True)
