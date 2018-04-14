import numpy as np

import torchrl.utils as U
from torchrl.agents import BaseAgent


class BatchAgent(BaseAgent):
    '''
    An agent that has methods for collecting a collection of trajectories.
    '''

    def generate_trajectories(self, steps_per_batch=-1, episodes_per_batch=-1):
        '''
        Generate a collection of trajectories, limited by
        ``timesteps_per_batch`` or ``episodes_per_batch``.

        Parameters
        ----------
        steps_per_batch: int
            Maximum number of time steps per batch
            (Default is -1, meaning it doesn't matter).
        episodes_per_batch: int
            Maximum number of episodes per batch
            (Default is -1, meaning it doesn't matter).

        Returns
        -------
        trajectories: list
            A list containing all sampled trajectories,
            each trajectory is in a ``dict``.
        '''
        assert steps_per_batch > -1 or episodes_per_batch > -1, \
        'You must define how many timesteps or episodes will be in each batch'

        total_steps = 0
        trajs = []

        while True:
            traj = self.env.run_one_episode(select_action_fn=self.select_action)
            trajs.append(traj)

            total_steps += len(traj['reward'])

            if (total_steps // steps_per_batch >= 1
                    or len(trajs) // episodes_per_batch >= 1):
                break

        return trajs

    def generate_batch(self, steps_per_batch, episodes_per_batch):
        trajs = self.generate_trajectories(steps_per_batch, episodes_per_batch)
        batch = U.Batch.from_trajs(trajs)

        return batch

    def train(self,
              steps_per_batch=-1,
              episodes_per_batch=-1,
              max_updates=-1,
              max_episodes=-1,
              max_steps=-1,
              **kwargs):
        super().train(
            max_updates=max_updates, max_episodes=max_episodes, max_steps=max_steps)

        while True:
            batch = self.generate_batch(steps_per_batch, episodes_per_batch)
            self.step(batch)

            self.write_logs()
            if self._check_termination():
                break
