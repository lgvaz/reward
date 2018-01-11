from torch.distributions import Categorical
from torchrl.models import VanillaPGModel
from torchrl.agents import BatchAgent


class VanillaPGAgent(BatchAgent):
    '''
    Vanilla Policy Gradient agent.
    '''
    _model = VanillaPGModel

    def train(self, num_episodes):
        # TODO: Use generate_trajectories
        for i_episode in range(num_episodes):
            batch = self.run_one_episode()
            self.model.train(batch)

            print(sum(batch['rewards']))
