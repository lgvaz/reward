r'''
Hyperparameters were choosed from `this <https://sourcegraph.com/github.com/Alfredvc/paac@a8bb993/-/blob/actor_learner.py?suggestion>`_ paper.
'''

import pdb
import torchrl.utils as U
from torchrl.agents import PGAgent
from torchrl.envs import AtariEnv, ParallelEnv
from torchrl.envs.gym_wrappers import atari_wrap
from torchrl.models import A2CModel, ValueModel
from torchrl.envs.wrappers import AtariWrapper
from torchrl.batchers import RolloutBatcher
from torchrl.runners import PAACRunner, SingleRunner
from torchrl.batchers.wrappers import CommonWraps

MAX_STEPS = 1.5e8
NUM_ENVS = 32
HORIZON = 5
# LR = 0.0007 * NUM_ENVS
LR = 1e-3

# Create environment
envs = [AtariWrapper(AtariEnv('PongNoFrameskip-v4')) for _ in range(NUM_ENVS)]
# env = ParallelEnv(envs)
# env = AtariWrapper(AtariEnv('PongNoFrameskip-v4'))
runner = PAACRunner(envs)
# runner = SingleRunner(envs[0])
batcher = CommonWraps.atari_wrap(RolloutBatcher(runner, batch_size=NUM_ENVS * HORIZON))

policy_model = A2CModel.from_arch(
    arch='a3c',
    batcher=batcher,
    entropy_coef=0.01,
    opt_params=dict(lr=LR, eps=1e-5),
    clip_grad_norm=0.5)

value_model = ValueModel.from_arch(
    arch='a3c',
    batcher=batcher,
    body=policy_model.body,
    num_epochs=1,
    num_mini_batches=1,
    opt_params=dict(lr=LR, eps=1e-5),
    clip_grad_norm=0.5,
    loss_coef=0.1)

# Create agent
agent = PGAgent(
    batcher=batcher,
    policy_model=policy_model,
    value_model=value_model,
    advantage=U.estimators.advantage.Baseline(gamma=0.99),
    vtarget=U.estimators.value.TDTarget(gamma=0.99),
    log_dir='logs/pong/nv/a2c-paac-separete-v2-2')

agent.train(max_steps=MAX_STEPS, log_freq=10)
