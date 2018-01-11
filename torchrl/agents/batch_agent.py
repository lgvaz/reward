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
        trajectories = []

        while True:
            trajectory = self.env.run_one_episode(select_action_fn=self.select_action)
            trajectories.append(trajectory)
            total_steps += trajectory['rewards'].shape[0]

            if (total_steps // timesteps_per_batch >= 1
                    or len(trajectories) // episodes_per_batch >= 1):
                break

        return trajectories
