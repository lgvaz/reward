import numpy as np

import torchrl.utils as U
from torchrl.agents import VanillaPGAgent


class ActorCriticAgent(VanillaPGAgent):
    '''
    Actor-critic agent.
    '''

    def add_to_trajectory(self, traj):
        self.model.add_state_values(traj)
        self.add_gae(traj)

    def add_gae(self, traj):
        advantages, vtarget = U.gae_estimation(
            rewards=traj['rewards'],
            state_values=traj['state_values'],
            gamma=self.gamma,
            gae_lambda=0.95)

        traj['advantages'] = advantages
        traj['vtarget'] = vtarget
