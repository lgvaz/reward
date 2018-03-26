import numpy as np

import torchrl.utils as U
from torchrl.agents import BatchAgent
from torchrl.models import ReinforceModel


class ReinforceAgent(BatchAgent):
    '''
    REINFORCE agent.
    '''
    _model = ReinforceModel

    def __init__(self,
                 env,
                 model=None,
                 gamma=0.99,
                 vtarget_mode='td_target',
                 normalize_advantages=True,
                 **kwargs):
        self.vtarget_mode = vtarget_mode
        self.normalize_advantages = normalize_advantages

        super().__init__(env=env, model=model, gamma=gamma, **kwargs)

    def add_to_batch(self, batch):
        if self.normalize_advantages:
            batch['advantages'] = U.normalize(batch['advantages'])

    def add_to_trajectory(self, traj):
        if self.model.value_nn is None:
            self.add_advantages_without_baseline(traj)
        else:
            self.add_advatanges_with_baseline(traj)

    def add_advantages_without_baseline(self, traj):
        returns = U.discounted_sum_rewards(traj['rewards'], self.gamma)
        traj['advantages'] = returns

    def add_advatanges_with_baseline(self, traj):
        self.model.add_state_values(traj)
        returns = U.discounted_sum_rewards(traj['rewards'], self.gamma)

        advantages = returns - traj['state_values']

        # Calculate target for the value network
        if self.vtarget_mode == 'complete_return':
            vtarget = returns
        elif self.vtarget_mode == 'td_target':
            vtarget = traj['rewards'] + self.gamma * np.append(traj['state_values'][1:],
                                                               0)
        else:
            raise ValueError('vtarget_mode {} not supported'.format(self.vtarget_mode))

        traj['advantages'] = advantages
        traj['vtarget'] = vtarget

    def train(self,
              timesteps_per_batch=-1,
              episodes_per_batch=-1,
              max_updates=-1,
              max_episodes=-1,
              max_steps=-1,
              **kwargs):
        super().train(
            max_updates=max_updates, max_episodes=max_episodes, max_steps=max_steps)

        while True:
            batch = self.generate_batch(timesteps_per_batch, episodes_per_batch)
            self.model.train(batch=batch, **kwargs)

            self.write_logs()
            if self._check_termination():
                break
