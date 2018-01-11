from torchrl.agents import BatchAgent
from torchrl.models import VanillaPGModel


class VanillaPGAgent(BatchAgent):
    '''
    Vanilla Policy Gradient agent.
    '''
    _model = VanillaPGModel

    def train(self, **kwargs):
        super().train(**kwargs)
        # TODO: Use generate_trajectories
        while True:
            batch = self.run_one_episode()
            self.model.train(batch)

            print(sum(batch['rewards']))

            if self._check_termination():
                break
