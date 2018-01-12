import numpy as np

from torchrl.agents import BaseAgent


class BatchAgent(BaseAgent):
    '''
    An agent that has methods for collecting a collection of trajectories.
    '''

    def generate_trajectories(self, timesteps_per_batch=-1, episodes_per_batch=-1):
        '''
        Generate a collection of trajectories, limited by
        ``timesteps_per_batch`` or ``episodes_per_batch``.

        Parameters
        ----------
        timesteps_per_batch: int
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
        assert timesteps_per_batch > -1 or episodes_per_batch > -1, \
        'You must define how many timesteps or episodes will be in each batch'

        total_steps = 0
        trajs = []

        while True:
            traj = self.env.run_one_episode(select_action_fn=self.select_action)
            self.add_to_trajectory(traj)
            trajs.append(traj)

            total_steps += traj['rewards'].shape[0]

            if (total_steps // timesteps_per_batch >= 1
                    or len(trajs) // episodes_per_batch >= 1):
                break

        return trajs

    def generate_batch(self, timesteps_per_batch, episodes_per_batch):
        trajs = self.generate_trajectories(timesteps_per_batch, episodes_per_batch)
        batch = self.concat_trajectories(trajs)
        self.add_to_batch(batch)

        return batch

    def concat_trajectories(self, trajs):
        batch = dict()
        for key in trajs[0]:
            batch[key] = np.concatenate([t[key] for t in trajs])

        return batch

    def add_to_trajectory(self, traj):
        pass

    def add_to_batch(self, batch):
        pass
