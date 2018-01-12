from torchrl.agents import BatchAgent
from torchrl.models import VanillaPGModel
from torchrl.utils import discounted_sum_rewards


class VanillaPGAgent(BatchAgent):
    '''
    Vanilla Policy Gradient agent.
    '''
    _model = VanillaPGModel

    def add_to_trajectory(self, traj):
        self.add_discounted_returns(traj)

    def add_discounted_returns(self, trajectory):
        discounted_returns = discounted_sum_rewards(trajectory['rewards'], self.gamma)
        trajectory['returns'] = discounted_returns

    def train(self, timesteps_per_batch=-1, episodes_per_batch=-1, **kwargs):
        super().train(**kwargs)
        while True:
            batch = self.generate_batch(timesteps_per_batch, episodes_per_batch)
            self.model.train(batch)

            self.write_logs()
            if self._check_termination():
                break

        import numpy as np
        return np.mean(self.env.rewards)
