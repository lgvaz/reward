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
        self.add_state_value(batch)
        self.add_advantage(batch)
        self.add_vtarget(batch)

        # Train models
        self.policy_model.train(batch)
        self.value_model.train(batch)

    def add_state_value(self, batch):
        if self.value_model is not None:
            batch.state_value = U.to_numpy(self.value_model(batch.state_t).view(-1))

    def add_advantage(self, batch):
        batch.advantage = self.advantage(batch)

    def add_vtarget(self, batch):
        batch.vtarget = self.vtarget(batch)

    # def add_advantages(self, batch):
    #     if self.advantages_mode == 'return':
    #         batch.advantage = batch.return_
    #     elif self.advantages_mode == 'baseline':
    #         batch.advantage = (batch.return_ - batch.state_value).float()
    #     elif self.advantages_mode == 'gae':
    #         # TODO: pass gae_lambda
    #         batch.advantage = U.gae_estimation(
    #             batch.reward, batch.done, batch.state_value, gamma=self.gamma)

    # def add_vtargets(self, batch):
    #     if self.vtarget_mode == 'return':
    #         batch.vtarget = batch.return_
    #     elif self.vtarget_mode == 'td_target':
    #         batch.vtarget = U.td_target(
    #             batch.reward, batch.done, batch.state_value, gamma=self.gamma)
    #     elif self.vtarget_mode == 'gae':
    #         assert self.advantages_mode == 'gae'
    #         batch.vtarget = batch.advantage + batch.state_value

    @classmethod
    def from_config(cls, config, env=None, **kwargs):
        if env is None:
            env = U.env_from_config(config)

        policy_nn_config = config.pop('policy_nn_config')
        value_nn_config = config.pop('value_nn_config', None)

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

        return cls(env, policy_nn, value_nn, **config.as_dict(), **kwargs)
