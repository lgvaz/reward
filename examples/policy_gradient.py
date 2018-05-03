import torch.nn as nn

import torchrl.utils as U
from torchrl.utils import Config
from torchrl.models import VanillaPGModel, SurrogatePGModel, PPOModel, ValueModel
from torchrl.agents import PGAgent
from torchrl.envs import GymEnv
from torchrl.envs import GymEnv, ParallelEnv
from torchrl.nn import ActionLinear

activation = nn.ReLU
# activation = nn.Tanh
# Define networks configs
# The input_features of the first layer will be automatically added.
policy_nn_config = Config(
    body=[
        dict(func=nn.Linear, out_features=64),
        dict(func=activation),
        dict(func=nn.Linear, in_features=64, out_features=64),
        dict(func=activation)
    ],
    head=[dict(func=ActionLinear)])
value_nn_config = Config(
    body=[
        dict(func=nn.Linear, out_features=64),
        dict(func=activation),
        dict(func=nn.Linear, in_features=64, out_features=64),
        dict(func=activation)
    ],
    head=[dict(func=nn.Linear, out_features=1)])

# Single env
# env = GymEnv('HalfCheetah-v2', normalize_states=True, scale_rewards=True)
# Parallel envs
envs = [GymEnv('Hopper-v2', normalize_states=True, scale_rewards=True) for _ in range(32)]
env = ParallelEnv(envs)

# TODO: actual method can't share bodies
policy_model_config = Config(nn_config=policy_nn_config)
policy_model = PPOModel.from_config(
    config=policy_model_config,
    env=env,
    opt_params=dict(lr=1e-3, eps=1e-5),
    clip_grad_norm=0.5)

value_model_config = Config(nn_config=value_nn_config)
value_model = ValueModel.from_config(
    config=value_model_config,
    env=env,
    opt_params=dict(lr=1e-3, eps=1e-5),
    clip_grad_norm=0.5)

# Create agent
agent = PGAgent(
    env,
    policy_model,
    value_model,
    # advantage=U.estimators.advantage.TD(gamma=0.99),
    # vtarget=U.estimators.value.TDTarget(gamma=0.99),
    log_dir='logs/hopper/parallel/32parallel-relu-2',
    normalize_advantages=True)
agent.train(max_steps=1e6, steps_per_batch=2048)
