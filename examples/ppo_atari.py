from torchrl.agents import PGAgent
from torchrl.envs import GymEnv, ParallelEnv
from torchrl.envs.gym_wrappers import atari_wrap
from torchrl.models import PPOClipModel, ValueModel
from torchrl.utils import piecewise_linear_schedule

MAX_STEPS = 15e6

# Create environment
envs = [
    GymEnv(
        'PongNoFrameskip-v4',
        fixed_normalize_states=True,
        clip_reward_range=1,
        wrappers=[atari_wrap]) for _ in range(8)
]
env = ParallelEnv(envs)

lr_schedule = piecewise_linear_schedule(
    values=[2.5e-4, 2.5e-4, 1e-4, 5e-5],
    boundaries=[MAX_STEPS * 0.1, MAX_STEPS * 0.5, MAX_STEPS * 0.8])

clip_schedule = piecewise_linear_schedule(
    values=[0.1, 0.1, 0.03], boundaries=[MAX_STEPS * 0.1, MAX_STEPS * 0.7])

policy_model = PPOClipModel.from_arch(
    arch='a3c',
    env=env,
    num_epochs=4,
    num_mini_batches=4,
    ppo_clip_range=clip_schedule,
    entropy_coef=0.01,
    opt_params=dict(lr=2.5e-4, eps=1e-5),
    lr_schedule=lr_schedule,
    clip_grad_norm=0.5)

value_model = ValueModel.from_arch(
    arch='a3c',
    env=env,
    body=policy_model.body,
    num_epochs=4,
    num_mini_batches=4,
    opt_params=dict(lr=2.5e-4, eps=1e-5),
    lr_schedule=lr_schedule,
    clip_range=clip_schedule,
    clip_grad_norm=0.5,
    loss_coef=0.5)

# Create agent
agent = PGAgent(
    env,
    policy_model=policy_model,
    value_model=value_model,
    log_dir='tests/pong/8parallel-p_e4_nmb4-0ent-v_nmb4_e4_vlc05_gc05_v4-3')

agent.train(max_steps=MAX_STEPS, steps_per_batch=128)
