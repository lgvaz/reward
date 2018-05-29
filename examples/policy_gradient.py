import torch.nn as nn

from torchrl.utils import Config
from torchrl.models import PPOClipModel, ValueModel
from torchrl.agents import PGAgent
from torchrl.envs import GymEnv, ParallelEnv
from torchrl.nn import ActionLinear

MAX_STEPS = 10e6

# activation = nn.ReLU
activation = nn.Tanh
# Define networks configs
policy_nn_config = Config(body=[
    dict(func=nn.Linear, out_features=64),
    dict(func=activation),
    dict(func=nn.Linear, in_features=64, out_features=64),
    dict(func=activation)
])
value_nn_config = Config(body=[
    dict(func=nn.Linear, out_features=64),
    dict(func=activation),
    dict(func=nn.Linear, in_features=64, out_features=64),
    dict(func=activation)
])

# Create environment
# env = GymEnv('HalfCheetah-v2', running_normalize_states=True, running_scale_rewards=True)
envs = [
    GymEnv('HalfCheetah-v2', running_normalize_states=True, running_scale_rewards=True)
    for _ in range(16)
]

env = ParallelEnv(envs)

policy_model_config = Config(nn_config=policy_nn_config)
policy_model = PPOClipModel.from_config(
    # policy_model = PPOAdaptiveModel.from_config(
    config=policy_model_config,
    env=env,
    # ppo_clip_range=piecewise_linear_schedule(
    #     values=[0.3, 0.3, 0.2, 0.2, 0.1],
    #     boundaries=[MAX_STEPS * 0.1, MAX_STEPS * 0.2, MAX_STEPS * 0.5, MAX_STEPS * 0.7]),
    num_epochs=10,
    num_mini_batches=1,
    opt_params=dict(lr=3e-4, eps=1e-5),
    # lr_schedule=piecewise_linear_schedule(
    #     values=[3e-4, 3e-4, 1e-4], boundaries=[MAX_STEPS * 0.1, MAX_STEPS * 0.5]),
    clip_grad_norm=None)

value_model_config = Config(nn_config=value_nn_config)
value_model = ValueModel.from_config(
    config=value_model_config,
    env=env,
    opt_params=dict(lr=3e-4, eps=1e-5),
    # lr_schedule=piecewise_linear_schedule(
    #     values=[3e-4, 3e-4, 1e-4], boundaries=[MAX_STEPS * 0.1, MAX_STEPS * 0.5]),
    num_mini_batches=8,
    num_epochs=10,
    clip_range=0.2,
    clip_grad_norm=None)

# Create agent
agent = PGAgent(
    env,
    policy_model=policy_model,
    value_model=value_model,
    log_dir='tests/cheetah/new/16parallel_p_nmb1_e10-v_nmb8_e10-cr02-gcNone-v9-1',
    normalize_advantages=True)
agent.train(max_steps=MAX_STEPS, steps_per_batch=2048)
