import numpy as np

import torchrl.utils as U
from torchrl.agents import BaseAgent
from torchrl.utils.estimators.estimation_funcs import discounted_sum_rewards


class BatchAgent(BaseAgent):
    '''
    An agent that has methods for collecting a collection of trajectories.
    '''

    def generate_trajectories(self):
        '''
        Generate a collection of trajectories, limited by
        ``timesteps_per_batch`` or ``episodes_per_batch``.

        Returns
        -------
        trajectories: list
            A list containing all sampled trajectories.
        '''
        assert self.steps_per_batch > -1 or self.episodes_per_batch > -1, \
        'You must define how many timesteps or episodes will be in each batch'
        assert not (self.steps_per_batch > -1 and self.episodes_per_batch > -1), \
        'You must define either steps or episodes per batch'

        if self.steps_per_batch > -1:
            trajs = self.env.run_n_steps(
                select_action_fn=self.select_action, num_steps=self.steps_per_batch)
        elif self.episodes_per_batch > -1:
            trajs = self.env.run_n_episodes(
                select_action_fn=self.select_action, num_episodes=self.episodes_per_batch)

        return trajs

    def train(self,
              *,
              steps_per_batch=-1,
              episodes_per_batch=-1,
              max_updates=-1,
              max_episodes=-1,
              max_steps=-1,
              log_freq=1):
        '''
        The main training loop.

        Parameters
        ----------
        steps_per_batch: int
            Maximum number of time steps per batch
            (Default is -1, meaning it doesn't matter).
        episodes_per_batch: int
            Maximum number of episodes per batch
            (Default is -1, meaning it doesn't matter).
        '''
        self.steps_per_batch = steps_per_batch
        self.episodes_per_batch = episodes_per_batch
        super().train(
            max_updates=max_updates,
            max_episodes=max_episodes,
            max_steps=max_steps,
            log_freq=log_freq)
