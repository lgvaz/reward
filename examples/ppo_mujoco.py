import reward as tr
import torch.nn as nn
from reward.agents import PGAgent
from reward.batchers import RolloutBatcher
from reward.envs import GymEnv
from reward.models import PPOClipModel, ValueClipModel
from reward.optimizers import JointOpt
from reward.runners import PAACRunner, SingleRunner
from reward.utils import Config
from reward.batchers.transforms import mujoco_transforms
import reward.batchers.transforms as tfms

MAX_STEPS = 40e6
# Create environment
envs = [GymEnv("Hopper-v2") for _ in range(16)]
runner = PAACRunner(envs)
# runner = SingleRunner(envs[0])

batcher = RolloutBatcher(
    runner, batch_size=2048, transforms=[tfms.StateRunNorm(), tfms.RewardRunScaler()]
)

# Create networks
actor_nn = tr.arch.MLP.from_env(
    env=envs[0], output_layer=tr.models.PPOClipModel.output_layer
)
critic_nn = tr.arch.MLP.from_env(
    env=envs[0], output_layer=tr.models.ValueClipModel.output_layer
)
# Create models
actor = tr.models.PPOClipModel(nn=actor_nn, batcher=batcher)
critic = tr.models.ValueClipModel(nn=critic_nn, batcher=batcher)

jopt = JointOpt(
    model=[actor, critic],
    epochs=4,
    mini_batches=4,
    opt_params=dict(lr=3e-4, eps=1e-5),
    clip_grad_norm=0.5,
)

# Create agent
agent = PGAgent(
    batcher=batcher,
    action_fn=actor.select_action,
    optimizer=jopt,
    actor=actor,
    critic=critic,
    normalize_advantages=True,
)
agent.train(max_steps=MAX_STEPS)
