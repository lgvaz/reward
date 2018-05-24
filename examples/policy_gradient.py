import torch.nn as nn

from torchrl.utils import Config
from torchrl.models import PPOClipModel, ValueModel
from torchrl.agents import PGAgent
from torchrl.envs import GymEnv, ParallelEnv
from torchrl.nn import ActionLinear

activation = nn.Tanh
# Define networks configs
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

envs = [
    GymEnv('HalfCheetah-v2', running_normalize_states=True, running_scale_rewards=True)
    for _ in range(16)
]
env = ParallelEnv(envs)

# TODO: actual method can't share bodies
policy_model_config = Config(nn_config=policy_nn_config)
policy_model = PPOClipModel.from_config(
    config=policy_model_config,
    env=env,
    opt_params=dict(lr=3e-4, eps=1e-5),
    clip_grad_norm=None)

value_model_config = Config(nn_config=value_nn_config)
value_model = ValueModel.from_config(
    config=value_model_config,
    env=env,
    opt_params=dict(lr=3e-4, eps=1e-5),
    batch_size=256,
    num_epochs=10,
    clip_range=0.2,
    clip_grad_norm=None)

# Create agent
agent = PGAgent(
    env,
    policy_model,
    value_model,
    log_dir=
    'logs/cheetah/new/16parallel-plr3e4_eps1e5-vlr3e4_b256_e10_eps1e5-gclipNone-v5-3',
    normalize_advantages=True)
agent.train(max_steps=10e6, steps_per_batch=2048)
