r"""
Hyperparameters were choosed from `this <https://sourcegraph.com/github.com/Alfredvc/paac@a8bb993/-/blob/actor_learner.py?suggestion>`_ paper.
"""

import torch
import torchrl.utils as U
from torchrl.agents import PGAgent
from torchrl.batchers import RolloutBatcher
from torchrl.envs import AtariEnv
from torchrl.envs.wrappers import AtariWrapper
from torchrl.models import A2CModel, ValueModel
from torchrl.optimizers import JointOpt
from torchrl.runners import PAACRunner
from torchrl.batchers.transforms import atari_transforms

MAX_STEPS = 1.5e8
NUM_ENVS = 32
HORIZON = 5
LR = 0.0007 * NUM_ENVS

# Create environment
envs = [AtariWrapper(AtariEnv("PongNoFrameskip-v4")) for _ in range(NUM_ENVS)]
runner = PAACRunner(envs)
batcher = RolloutBatcher(
    runner, batch_size=HORIZON * NUM_ENVS, transforms=atari_transforms()
)

policy_model = A2CModel.from_arch(arch="a3c", batcher=batcher, entropy_coef=0.02)

value_model = ValueModel.from_arch(arch="a3c", batcher=batcher, body=policy_model.body)

opt = JointOpt(
    [policy_model, value_model],
    opt_fn=torch.optim.RMSprop,
    opt_params=dict(lr=LR, eps=1e-1, centered=False),
    clip_grad_norm=3.,
    loss_coef=[5., 5. * 0.25],
)

# Create agent
agent = PGAgent(
    batcher=batcher,
    optimizer=opt,
    policy_model=policy_model,
    value_model=value_model,
    advantage=U.estimators.advantage.Baseline(gamma=0.99),
    vtarget=U.estimators.value.CompleteReturn(gamma=0.99),
    normalize_advantages=False,
    log_dir="tests/pong/nv/paper-v6-4",
)

agent.train(max_steps=MAX_STEPS, log_freq=10)
