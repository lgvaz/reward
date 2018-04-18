import torch.nn as nn

import torchrl.utils as U
from torchrl.utils import Config
from torchrl.models import VanillaPGModel, SurrogatePGModel, PPOModel, ValueModel
from torchrl.agents import PGAgent
from torchrl.envs import GymEnv
from torchrl.nn import ActionLinear

# Define networks configs
policy_nn_config = Config(
    body=[
        dict(func=nn.Linear, out_features=64),
        dict(func=nn.ReLU),
        dict(func=nn.Linear, in_features=64, out_features=64),
        dict(func=nn.ReLU)
    ],
    head=[dict(func=ActionLinear)])
value_nn_config = Config(
    body=[
        dict(func=nn.Linear, out_features=64),
        dict(func=nn.ReLU),
        dict(func=nn.Linear, in_features=64, out_features=64),
        dict(func=nn.ReLU)
    ],
    head=[dict(func=nn.Linear, out_features=1)])

# # Define networks configs
# policy_nn_config = Config(
#     body=[
#         dict(func=nn.Linear, out_features=64),
#         dict(func=nn.Tanh),
#         dict(func=nn.Linear, in_features=64, out_features=64),
#         dict(func=nn.Tanh)
#     ],
#     head=[dict(func=ActionLinear)])
# value_nn_config = Config(
#     body=[
#         dict(func=nn.Linear, out_features=64),
#         dict(func=nn.Tanh),
#         dict(func=nn.Linear, in_features=64, out_features=64),
#         dict(func=nn.Tanh)
#     ],
#     head=[dict(func=nn.Linear, out_features=1)])

# Create environment
# env = GymEnv('InvertedPendulum-v1', normalize_states=False)
# env = GymEnv('InvertedDoublePendulum-v1', normalize_states=False)
# env = GymEnv('Pendulum-v0', normalize_states=False)
env = GymEnv('Hopper-v1', normalize_states=True)
# env = GymEnv('HalfCheetah-v1', normalize_states=True)
# env = GymEnv('CartPole-v0', normalize_states=False)
# env_config = Config(func=GymEnv, env_name='Pendulum-v0', normalize_states=False)

# TODO: actual method can't share bodies
policy_model_config = Config(nn_config=policy_nn_config)
policy_model = PPOModel.from_config(
    config=policy_model_config,
    env=env,
    opt_params=dict(lr=1e-3, eps=1e-5),
    clip_grad_norm=0.5)

value_model_config = Config(nn_config=value_nn_config)
value_model = ValueModel.from_config(
    config=value_model_config, env=env, opt_params=dict(lr=1e-3))

# Create agent
agent = PGAgent(
    env,
    policy_model,
    value_model,
    # log_dir='logs/hopper/relu-1e3lr-gradnorm-v1',
    log_dir='tests/hopper/relu-1e3lr-gradnorm-v1',
    normalize_advantages=True)
agent.train(max_steps=1e6, steps_per_batch=2048)
