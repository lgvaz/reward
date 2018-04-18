import numpy as np
import torchrl.utils as U
from torchrl.agents import BatchAgent
from torchrl.models import BasePGModel, ValueModel


class PGAgent(BatchAgent):
    def __init__(self,
                 env,
                 policy_model,
                 value_model=None,
                 normalize_advantages=True,
                 advantage=U.estimators.advantage.GAE(gamma=0.99, gae_lambda=0.95),
                 vtarget=U.estimators.value.GAE(),
                 **kwargs):
        super().__init__(env, **kwargs)

        self.normalize_advantages = normalize_advantages
        self.advantage = advantage
        self.vtarget = vtarget

        self._register_model('policy', policy_model)
        self._register_model('value', value_model)

    def step(self):
        batch = self.generate_batch(self.steps_per_batch, self.episodes_per_batch)

        self.add_state_value(batch)
        self.add_advantage(batch)
        self.add_vtarget(batch)
        if self.normalize_advantages:
            batch.advantage = U.normalize(batch.advantage)

        self.models.policy.train(batch)
        self.models.value.train(batch)

    def add_state_value(self, batch):
        if self.models.value is not None:
            batch.state_value = U.to_numpy(self.models.value(batch.state_t).view(-1))

    def add_advantage(self, batch):
        batch.advantage = self.advantage(batch)

    def add_vtarget(self, batch):
        batch.vtarget = self.vtarget(batch)

    # TODO: Reimplement this method
    @classmethod
    def from_config(cls, config, env=None, policy_model_class=None, **kwargs):
        if env is None:
            env = U.env_from_config(config)

        # If the policy_model_class is given it should overwrite key from config
        if policy_model_class is not None:
            config.pop('policy_model_class')
        else:
            policy_model_class = config.pop('policy_model_class')

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

        return cls(
            env=env,
            policy_model_class=policy_model_class,
            policy_nn=policy_nn,
            value_nn=value_nn,
            **config.as_dict(),
            **kwargs)
