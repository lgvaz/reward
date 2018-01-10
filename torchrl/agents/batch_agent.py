from torchrl.agents import BaseAgent


class BatchAgent(BaseAgent):
    def generate_trajectories(self,
                              timesteps_per_batch=-1,
                              episodes_per_batch=-1,
                              **kwargs):
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
