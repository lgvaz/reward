import os

import torch.nn as nn

import torchrl.utils as U
from bench import MUJOCO_SIMPLE_BENCH, ROBOSCHOOL_SIMPLE_BENCH, task_gen
from torchrl.agents import PGAgent
from torchrl.envs import GymEnv
from torchrl.models import PPOModel, ValueModel
from torchrl.nn import ActionLinear
from torchrl.utils import Config
from utils import config2str

CONFIG = Config(
    activation=nn.ReLU,
    max_steps=1e6,
    steps_per_batch=2048,
    episodes_per_batch=-1,
    normalize_advantages=True,
    normalize_states=True,
    scale_rewards=True,
    advantage=U.estimators.advantage.GAE(gamma=0.99, gae_lambda=0.95),
    policy_opt_params=dict(lr=1e-3, eps=1e-5),
    value_opt_params=dict(lr=1e-3, eps=1e-5),
    clip_grad_norm=0.5)

TASK = MUJOCO_SIMPLE_BENCH
NUM_WORKERS = 5


def run_bench(config):
    CONFIG.update(config)
    log_dir = config2str(CONFIG)
    log_dir = 'PPO-' + log_dir
    log_dir = os.path.join('logs', CONFIG.env_name, log_dir)

    # Create env
    env = GymEnv(
        env_name=CONFIG.env_name,
        normalize_states=CONFIG.normalize_states,
        scale_rewards=CONFIG.scale_rewards)

    # Define networks configs
    policy_nn_config = Config(
        body=[
            dict(func=nn.Linear, out_features=64),
            dict(func=CONFIG.activation),
            dict(func=nn.Linear, in_features=64, out_features=64),
            dict(func=CONFIG.activation)
        ],
        head=[dict(func=ActionLinear)])
    value_nn_config = Config(
        body=[
            dict(func=nn.Linear, out_features=64),
            dict(func=CONFIG.activation),
            dict(func=nn.Linear, in_features=64, out_features=64),
            dict(func=CONFIG.activation)
        ],
        head=[dict(func=nn.Linear, out_features=1)])

    # Create Models
    policy_model_config = Config(nn_config=policy_nn_config)
    policy_model = PPOModel.from_config(
        config=policy_model_config,
        env=env,
        opt_params=CONFIG.policy_opt_params,
        clip_grad_norm=CONFIG.clip_grad_norm)

    value_model_config = Config(nn_config=value_nn_config)
    value_model = ValueModel.from_config(
        config=value_model_config, env=env, opt_params=CONFIG.value_opt_params)

    # Create agent
    agent = PGAgent(
        env=env,
        policy_model=policy_model,
        value_model=value_model,
        advantage=CONFIG.advantage,
        normalize_advantages=CONFIG.normalize_advantages,
        log_dir=log_dir)

    agent.train(
        max_steps=CONFIG.max_steps,
        steps_per_batch=CONFIG.steps_per_batch,
        episodes_per_batch=CONFIG.episodes_per_batch)


if __name__ == '__main__':
    import multiprocessing
    p = multiprocessing.Pool(NUM_WORKERS)
    p.map_async(run_bench, task_gen(TASK)).get()
