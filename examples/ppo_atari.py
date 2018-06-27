from torchrl.agents import PGAgent
from torchrl.batchers import RolloutBatcher
from torchrl.batchers.wrappers import CommonWraps
from torchrl.envs import AtariEnv
from torchrl.envs.wrappers import AtariWrapper
from torchrl.models import PPOClipModel, ValueClipModel
from torchrl.runners import PAACRunner, SingleRunner
from torchrl.utils import piecewise_linear_schedule
from torchrl.optimizers import SingleOpt, JointOpt

MAX_STEPS = 50e6
HORIZON = 128
NUM_ENVS = 8

# Create environment
envs = [AtariWrapper(AtariEnv('PongNoFrameskip-v4')) for _ in range(NUM_ENVS)]
runner = PAACRunner(envs)
# runner = SingleRunner(envs[0])
batcher = CommonWraps.atari_wrap(RolloutBatcher(runner, batch_size=HORIZON * NUM_ENVS))

lr_schedule = piecewise_linear_schedule(
    values=[4e-4, 4e-4, 1e-4, 5e-5],
    boundaries=[MAX_STEPS * 0.1, MAX_STEPS * 0.5, MAX_STEPS * 0.8])

clip_schedule = piecewise_linear_schedule(
    values=[0.1, 0.1, 0.03], boundaries=[MAX_STEPS * 0.1, MAX_STEPS * 0.7])

policy_model = PPOClipModel.from_arch(
    arch='a3c',
    batcher=batcher,
    num_epochs=4,
    num_mini_batches=4,
    # ppo_clip_range=clip_schedule,
    entropy_coef=0.01,
    opt_params=dict(lr=2.5e-4, eps=1e-5),
    # lr_schedule=lr_schedule,
    clip_grad_norm=0.5)

value_model = ValueClipModel.from_arch(
    arch='a3c',
    batcher=batcher,
    body=policy_model.body,
    num_epochs=4,
    num_mini_batches=4,
    opt_params=dict(lr=2.5e-4, eps=1e-5),
    # lr_schedule=lr_schedule,
    clip_range=0.1,
    # clip_range=clip_schedule,
    clip_grad_norm=0.5,
    loss_coef=0.5)

opt = JointOpt(
    model=[policy_model, value_model],
    num_epochs=4,
    num_mini_batches=4,
    opt_params=dict(lr=3e-4, eps=1e-5),
    clip_grad_norm=0.5,
    loss_coef=[1., 0.5])

# Create agent
agent = PGAgent(
    batcher=batcher,
    optimizer=opt,
    policy_model=policy_model,
    value_model=value_model,
    log_dir='logs/pong/nv/paper-nv2-v2-0')

agent.train(max_steps=MAX_STEPS, log_freq=10)
