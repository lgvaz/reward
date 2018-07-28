import torch.nn as nn

import torchrl.utils as U
from torchrl.envs import GymEnv
from torchrl.runners import SingleRunner, PAACRunner
from torchrl.batchers import RolloutBatcher
from torchrl.models import VanillaPGModel, ValueModel
from torchrl.optimizers import SingleOpt, JointOpt, ListOpt
from torchrl.agents import PGAgent

from torchrl.envs.wrappers.record_wrappers import GymRecorder

# Define networks configs
policy_nn_config = U.Config(
    body=[dict(func=nn.Linear, out_features=64), dict(func=nn.Tanh)]
)

value_nn_config = U.Config(
    body=[dict(func=nn.Linear, out_features=64), dict(func=nn.Tanh)]
)

# Create env and runner
env = GymRecorder(GymEnv("CartPole-v1"), directory="videos")
runner = SingleRunner(env)

# Create parallel envs
# envs = [GymEnv('CartPole-v1') for _ in range(16)]
# runner = PAACRunner(envs)

# Create batcher
batcher = RolloutBatcher(runner, batch_size=512)

# Create models
policy_model_config = U.Config(nn_config=policy_nn_config)
policy_model = VanillaPGModel.from_config(
    config=policy_model_config, batcher=batcher, entropy_coef=0.01
)
pot = SingleOpt(policy_model, opt_params=dict(lr=1e-3))

value_model_config = U.Config(nn_config=value_nn_config)
value_model = ValueModel.from_config(config=value_model_config, batcher=batcher)
vot = SingleOpt(
    value_model, num_epochs=10, num_mini_batches=4, opt_params=dict(lr=1e-3)
)

opt = ListOpt(optimizers=[pot, vot])
# opt = JointOpt(model=[policy_model, value_model])

# Create agent
agent = PGAgent(
    batcher=batcher,
    optimizer=opt,
    policy_model=policy_model,
    value_model=value_model,
    # advantage=U.estimators.advantage.Baseline(gamma=0.99),
    # vtarget=U.estimators.value.CompleteReturn(gamma=0.99),
    log_dir="tests/vanilla/cp/1p-opt-v8-0",
)

agent.train(max_steps=1e6, log_freq=10)
