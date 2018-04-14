import numpy as np
import torchrl.utils as U
from torchrl.agents import BatchAgent
from torchrl.models import ValueModel, PGModel


class BasePGAgent(BatchAgent):
    def __init__(self, env, policy_nn, value_nn=None, vtarget_mode='td_target', **kwargs):
        self.policy_nn = policy_nn
        self.value_nn = value_nn
        self.vtarget_mode = vtarget_mode

        super().__init__(env, **kwargs)

    def create_models(self):
        self.policy_model = PGModel(self.policy_nn, self.env.action_info)

        if self.value_nn is not None:
            self.value_model = ValueModel(self.value_nn)
        else:
            self.value_model = None

    def step(self, batch):
        self.add_returns(batch)
        self.add_state_values(batch)
        self.add_vtargets(batch)
        self.add_advantages(batch)

        # Train models
        self.policy_model.train(batch)
        self.value_model.train(batch)

    def add_returns(self, batch):
        batch.returns = U.discounted_sum_rewards(batch.rewards, batch.dones, self.gamma)

    def add_state_values(self, batch):
        if self.value_model is not None:
            batch.state_values = self.value_model(batch.state_ts).view(-1).detach()

    def add_vtargets(self, batch):
        # TODO: More vtargets modes
        if self.vtarget_mode == 'td_target':
            batch.vtargets = batch.rewards + (
                1 - batch.dones) * self.gamma * np.append(batch.state_values[1:], 0)

        elif self.vtarget_mode == 'return':
            batch.vtargets = batch.returns

    def add_advantages(self, batch):
        if self.value_model is not None:
            batch.advantages = (batch.returns - batch.state_values).float()
        # batch.advantages = batch.returns

    @classmethod
    def from_config(cls, config, env=None):
        if env is None:
            env = U.env_from_config(config)

        policy_nn_config = config.get('policy_nn_config')
        value_nn_config = config.get('value_nn_config')

        policy_nn = U.nn_from_config(policy_nn_config, env.state_info, env.action_info)
        if value_nn_config is not None:
            if value_nn_config.get('body') is None:
                print('Policy NN and Value NN are sharing bodies')
                value_nn_body = policy_nn.layers[0]
            else:
                print('Policy NN and Value NN are using different bodies')
                value_nn_body = None
            value_nn = U.nn_from_config(
                value_nn_config, env.state_info, env.action_info, body=value_nn_body)
        else:
            value_nn = None

        return cls(env, policy_nn, value_nn)
