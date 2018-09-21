import torch.nn as nn
import reward as tr
import reward.utils as U


env = tr.envs.GymEnv("HalfCheetah-v2")
# envs = [tr.envs.GymEnv("InvertedPendulum-v2") for _ in range(8)]

# Define networks
actor_nn = tr.arch.MLP.from_env(
    env=env,
    output_layer=tr.models.DDPGActor.output_layer,
    hidden=[64],
    activation=nn.ReLU,
)
critic_nn = tr.arch.DDPGCritic.from_env(
    env=env,
    output_layer=tr.models.DDPGCritic.output_layer,
    before=[64],
    activation=nn.ReLU,
)

runner = tr.runners.SingleRunner(env)

batcher = tr.batchers.ReplayBatcher(
    runner=runner,
    batch_size=64,
    replay_buffer_maxlen=50e3,
    learning_freq=1,
    transforms=tr.batchers.transforms.mujoco_transforms(),
)

critic = tr.models.DDPGCritic(
    nn=critic_nn, batcher=batcher, target_up_freq=1, target_up_weight=.001
)

actor = tr.models.DDPGActor(
    nn=actor_nn, batcher=batcher, critic=critic, target_up_freq=1, target_up_weight=.001
)

opt_critic = tr.optimizers.SingleOpt(model=critic, opt_params=dict(lr=1e-3))
opt_actor = tr.optimizers.SingleOpt(model=actor, opt_params=dict(lr=1e-4))
opt = tr.optimizers.ListOpt([opt_critic, opt_actor])

agent = tr.agents.DDPGAgent(
    actor=actor,
    critic=critic,
    batcher=batcher,
    action_fn=actor.select_action,
    optimizer=opt,
)

agent.train(max_steps=10e6, log_freq=500)
