import torchrl.utils as U
from torchrl.agents import BatchAgent
from torchrl.models import ValueModel, PGModel


class BasePGAgent(BatchAgent):
    def __init__(self, env, policy_nn, value_nn=None, **kwargs):
        self.policy_nn = policy_nn
        self.value_nn = value_nn

        super().__init__(env, **kwargs)

    def create_models(self):
        self.policy_model = PGModel(self.policy_nn, self.env.action_info)
        self.value_model = ValueModel(self.value_nn)

    def step(self, batch):
        self.add_returns(batch)
        self.add_vtargets(batch)
        self.add_advantages(batch)

        # Train models
        self.policy_model.train(batch)
        self.value_model.train(batch)

    def train(self,
              steps_per_batch=-1,
              episodes_per_batch=-1,
              max_updates=-1,
              max_episodes=-1,
              max_steps=-1,
              **kwargs):
        super().train(
            max_updates=max_updates, max_episodes=max_episodes, max_steps=max_steps)

        while True:
            batch = self.generate_batch(steps_per_batch, episodes_per_batch)
            self.step(batch)

            self.write_logs()
            if self._check_termination():
                break

    def add_returns(self, batch):
        batch.returns = U.discounted_sum_rewards(batch.rewards, batch.dones, self.gamma)

    def add_vtargets(self, batch):
        # TODO: More vtargets modes
        batch.vtargets = batch.returns

    def add_advantages(self, batch):
        batch.advantages = batch.returns

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
