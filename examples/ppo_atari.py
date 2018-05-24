import torch.nn as nn

from torchrl.agents import PGAgent
from torchrl.envs import GymEnv, ParallelEnv
from torchrl.envs.gym_wrappers import atari_wrap
from torchrl.models import PPOClipModel, ValueModel
from torchrl.nn import ActionLinear, FlattenLinear
from torchrl.utils import Config, piecewise_linear_schedule

MAX_STEPS = 40e6

activation = nn.ReLU
# Define networks configs
policy_nn_config = Config(
    body=[
        dict(func=nn.Conv2d, out_channels=16, kernel_size=8, stride=4),
        dict(func=nn.Conv2d, in_channels=16, out_channels=32, kernel_size=4, stride=2)
    ],
    head=[
        dict(func=FlattenLinear, out_features=256),
        dict(func=ActionLinear, in_features=256)
    ])
# Will share body with policy_nn
value_nn_config = Config(head=[
    dict(func=FlattenLinear, out_features=256),
    dict(func=nn.Linear, in_features=256, out_features=1)
])

# Create environment
envs = [
    GymEnv(
        'PongNoFrameskip-v4',
        fixed_normalize_states=True,
        clip_reward_range=1,
        wrappers=[atari_wrap]) for _ in range(16)
]
env = ParallelEnv(envs)

policy_model_config = Config(nn_config=policy_nn_config)
policy_model = PPOClipModel.from_config(
    config=policy_model_config,
    env=env,
    num_epochs=4,
    ppo_clip_range=0.1,
    entropy_coef=0.01,
    opt_params=dict(lr=3e-4, eps=1e-5),
    # lr_schedule=piecewise_linear_schedule(
    #     values=[5e-4, 3e-4, 3e-4, 1e-4],
    #     boundaries=[MAX_STEPS * 0.1, MAX_STEPS * 0.5, MAX_STEPS * 0.7]),
    # # lr_schedule=piecewise_linear_schedule(
    #     values=[3e-4, 3e-4, 1e-4, 5e-5],
    #     boundaries=[MAX_STEPS * 0.1, MAX_STEPS * 0.5, MAX_STEPS * 0.8]),
    clip_grad_norm=0.5)

value_model_config = Config(nn_config=value_nn_config)
value_model = ValueModel.from_config(
    config=value_model_config,
    env=env,
    body=policy_model.body,
    opt_params=dict(lr=3e-4, eps=1e-5),
    # lr_schedule=piecewise_linear_schedule(
    #     values=[5e-4, 3e-4, 3e-4, 1e-4],
    #     boundaries=[MAX_STEPS * 0.1, MAX_STEPS * 0.5, MAX_STEPS * 0.7]),
    # lr_schedule=piecewise_linear_schedule(
    #     values=[3e-4, 3e-4, 1e-4, 5e-5],
    #     boundaries=[MAX_STEPS * 0.1, MAX_STEPS * 0.5, MAX_STEPS * 0.8]),
    batch_size=256,
    num_epochs=4,
    clip_range=0.1,
    clip_grad_norm=0.5,
    loss_coef=0.1)

# Create agent
agent = PGAgent(
    env,
    policy_model,
    value_model,
    log_dir='logs/pong/16parallel-p_e4_eps1e5-v_3e4_b256_e4_eps1e5-gc05-v2-0',
    normalize_advantages=True)
agent.train(max_steps=MAX_STEPS, steps_per_batch=2048)
