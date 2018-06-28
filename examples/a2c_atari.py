r'''
Hyperparameters were choosed from `this <https://sourcegraph.com/github.com/Alfredvc/paac@a8bb993/-/blob/actor_learner.py?suggestion>`_ paper.
'''

from torchrl.agents import PGAgent
from torchrl.batchers import RolloutBatcher
from torchrl.batchers.wrappers import CommonWraps
from torchrl.envs import AtariEnv
from torchrl.envs.wrappers import AtariWrapper
from torchrl.models import A2CModel, ValueModel
from torchrl.optimizers import JointOpt
from torchrl.runners import PAACRunner

MAX_STEPS = 1.5e8
NUM_ENVS = 32
HORIZON = 5
LR = 0.0007 * NUM_ENVS

# Create environment
envs = [AtariWrapper(AtariEnv('PongNoFrameskip-v4')) for _ in range(NUM_ENVS)]
runner = PAACRunner(envs)
batcher = RolloutBatcher(runner, batch_size=NUM_ENVS * HORIZON)
batcher = CommonWraps.atari_wrap(batcher)

policy_model = A2CModel.from_arch(arch='a3c', batcher=batcher, entropy_coef=0.01)

value_model = ValueModel.from_arch(arch='a3c', batcher=batcher, body=policy_model.body)

opt = JointOpt(
    [policy_model, value_model],
    opt_params=dict(lr=LR, eps=1e-5),
    clip_grad_norm=40,
    loss_coef=[1., 0.5])

# Create agent
agent = PGAgent(
    batcher=batcher,
    optimizer=opt,
    policy_model=policy_model,
    value_model=value_model,
    # advantage=U.estimators.advantage.Baseline(gamma=0.99),
    # vtarget=U.estimators.value.TDTarget(gamma=0.99),
    log_dir='logs/pong/nv/a2c-paac-v3-0')

agent.train(max_steps=MAX_STEPS, log_freq=10)
