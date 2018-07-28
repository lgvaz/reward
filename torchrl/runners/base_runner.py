from abc import ABC, abstractmethod

import numpy as np

import torchrl.utils as U


class BaseRunner(ABC):
    def __init__(self, env):
        self.env = env
        self.rewards = []
        self.num_steps = 0
        self.ep_lengths = []
        self.new_eps = 0
        self._last_logged_ep = 0

    @property
    @abstractmethod
    def num_envs(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self, action):
        pass

    @abstractmethod
    def get_state_info(self):
        pass

    @abstractmethod
    def get_action_info(self):
        pass

    # @property
    # def rewards(self):
    #     return self._rewards

    # @property
    # def num_steps(self):
    #     return self._steps

    def close(self):
        raise NotImplementedError

    @property
    def num_episodes(self):
        return len(self.rewards)

    def evaluate(self, env, select_action_fn, state_transform, logger):
        print("".join([22 * "-", " Running Evaluation ", 22 * "-"]))

        state = env.reset()[None]
        state = state_transform(state, training=False)
        traj = U.memories.SimpleMemory(initial_keys=["rewards"])
        traj.length = 0

        done = False
        while not done:
            action = select_action_fn(state=state)
            next_state, reward, done, info = env.step(action)

            state = next_state[None]
            state = state_transform(state, training=False)

            traj.rewards.append(reward)
            traj.length += 1

        logger.add_log("Env/Reward/Evaluation", np.sum(traj.rewards))
        logger.add_log("Env/Length/Evaluation", traj.length)

    def write_logs(self, logger):
        new_eps = abs(self._last_logged_ep - self.num_episodes)
        if new_eps != 0:
            self.new_eps = new_eps
            self._last_logged_ep = self.num_episodes

        logger.add_log(
            "Env/Reward/Episode (New)", np.mean(self.rewards[-self.new_eps :])
        )
        logger.add_log("Env/Reward/Episode (Last 50)", np.mean(self.rewards[-50:]))
        logger.add_log("Env/Length/Episode (New)", np.mean(self.ep_lengths[-new_eps:]))
        logger.add_log("Env/Length/Episode (Last 50)", np.mean(self.ep_lengths[-50:]))
