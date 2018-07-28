import numpy as np
import torchrl.utils as U
from torchrl.agents import BaseAgent


# TODO: docstring
class PGAgent(BaseAgent):
    """
    Policy Gradient Agent, compatible with all PG models.

    This agent encapsulates a policy_model and optionally a value_model,
    it defines the steps needed for the training loop (see :meth:`step`),
    and calculates all the necessary values to train the model(s).

    Parameters
    ----------
    env: torchrl.envs
        A torchrl environment.
    policy_model: torchrl.models
        Should be a subclass of ``torchrl.models.BasePGModel``
    value_model: torchrl.models
        Should be an instance of ``torchrl.models.ValueModel`` (Default is None)
    normalize_advantages: bool
        If True, normalize the advantages per batch.
    advantage: torchrl.utils.estimators.advantage
        Class used for calculating the advantages.
    vtarget: torchrl.utils.estimators.value
        Class used for calculating the states target values.
    """

    def __init__(
        self,
        batcher,
        *,
        policy_model,
        value_model=None,
        normalize_advantages=True,
        advantage=U.estimators.advantage.GAE(gamma=0.99, gae_lambda=0.95),
        vtarget=U.estimators.value.FromAdvantage(),
        **kwargs
    ):
        super().__init__(batcher=batcher, **kwargs)

        self.normalize_advantages = normalize_advantages
        self.advantage = advantage
        self.vtarget = vtarget

        self._register_model("policy", policy_model)
        if value_model is not None:
            self._register_model("value", value_model)

    def step(self):
        batch = self.generate_batch()

        self.add_state_value(batch)
        self.add_advantage(batch)
        self.add_vtarget(batch)

        batch = batch.concat_batch()

        if self.normalize_advantages:
            batch.advantage = U.normalize(batch.advantage)

        self.train_models(batch)

    def add_state_value(self, batch):
        if self.models.value is not None:
            s = batch.state_t_and_tp1
            v = self.models.value(s.reshape(-1, *s.shape[2:]))

            v = U.to_np(v).reshape(s.shape[:2])
            batch.state_value_t_and_tp1 = v
            batch.state_value_t = v[:-1]
            batch.state_value_tp1 = v[1:]

    def add_advantage(self, batch):
        batch.advantage = self.advantage(batch)

    def add_vtarget(self, batch):
        batch.vtarget = self.vtarget(batch)

    # TODO: Reimplement this method
    # @classmethod
    # def from_config(cls, config, env=None, policy_model_class=None, **kwargs):
    #     if env is None:
    #         env = U.env_from_config(config)

    #     # If the policy_model_class is given it should overwrite key from config
    #     if policy_model_class is not None:
    #         config.pop('policy_model_class')
    #     else:
    #         policy_model_class = config.pop('policy_model_class')

    #     policy_nn_config = config.pop('policy_nn_config')
    #     value_nn_config = config.pop('value_nn_config', None)

    #     policy_nn = U.nn_from_config(policy_nn_config, env.get_state_info(), env.get_action_info())
    #     if value_nn_config is not None:
    #         if value_nn_config.get('body') is None:
    #             print('Policy NN and Value NN are sharing bodies')
    #             value_nn_body = policy_nn.layers[0]
    #         else:
    #             print('Policy NN and Value NN are using different bodies')
    #             value_nn_body = None
    #         value_nn = U.nn_from_config(
    #             value_nn_config, env.get_state_info(), env.get_action_info(), body=value_nn_body)
    #     else:
    #         value_nn = None

    #     return cls(
    #         env=env,
    #         policy_model_class=policy_model_class,
    #         policy_nn=policy_nn,
    #         value_nn=value_nn,
    #         **config.as_dict(),
    #         **kwargs)
