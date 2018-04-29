import os

import torch.nn as nn

import torchrl.utils as U
from bench import MUJOCO_SIMPLE_BENCH, MUJOCO_ESSENTIAL_BENCH, task_gen
from torchrl.agents import PGAgent
from torchrl.envs import GymEnv
from torchrl.models import PPOModel, ValueModel
from torchrl.nn import ActionLinear
from torchrl.utils import Config
from utils import config2str

CONFIG = Config(
    activation=nn.Tanh,
    max_steps=1e6,
    steps_per_batch=2048,
    episodes_per_batch=-1,
    normalize_advantages=True,
    normalize_states=True,
    scale_rewards=True,
    advantage=U.estimators.advantage.GAE(gamma=0.99, gae_lambda=0.95),
    policy_opt_params=dict(lr=1e-3, eps=1e-5),
    value_opt_params=dict(lr=1e-3, eps=1e-5),
    clip_grad_norm=None)

TASK = MUJOCO_ESSENTIAL_BENCH
NUM_WORKERS = None


def run_bench(config):
    # CONFIG.update(config)
    log_dir = config2str(config)
    log_dir = 'TEST/PPO-10epochv' + log_dir
    log_dir = os.path.join('logs', config.env_name, log_dir)

    # Create env
    env = GymEnv(
        env_name=config.env_name,
        normalize_states=config.normalize_states,
        scale_rewards=config.scale_rewards)

    # Define networks configs
    policy_nn_config = Config(
        body=[
            dict(func=nn.Linear, out_features=64),
            dict(func=config.activation),
            dict(func=nn.Linear, in_features=64, out_features=64),
            dict(func=config.activation)
        ],
        head=[dict(func=ActionLinear)])
    value_nn_config = Config(
        body=[
            dict(func=nn.Linear, out_features=64),
            dict(func=config.activation),
            dict(func=nn.Linear, in_features=64, out_features=64),
            dict(func=config.activation)
        ],
        head=[dict(func=nn.Linear, out_features=1)])

    # Create Models
    policy_model_config = Config(nn_config=policy_nn_config)
    policy_model = PPOModel.from_config(
        config=policy_model_config,
        env=env,
        opt_params=config.policy_opt_params,
        clip_grad_norm=config.clip_grad_norm)

    value_model_config = Config(nn_config=value_nn_config)
    value_model = ValueModel.from_config(
        config=value_model_config, env=env, opt_params=config.value_opt_params)

    # Create agent
    agent = PGAgent(
        env=env,
        policy_model=policy_model,
        value_model=value_model,
        advantage=config.advantage,
        normalize_advantages=config.normalize_advantages,
        log_dir=log_dir)

    agent.train(
        max_steps=config.max_steps,
        steps_per_batch=config.steps_per_batch,
        episodes_per_batch=config.episodes_per_batch)


if __name__ == '__main__':
    import multiprocessing
    p = multiprocessing.Pool(NUM_WORKERS)

    setattr(CONFIG, 'activation', nn.Tanh)
    setattr(CONFIG, 'policy_opt_params', dict(lr=3e-4, eps=1e-5))
    setattr(CONFIG, 'value_opt_params', dict(lr=3e-4, eps=1e-5))
    p.map_async(run_bench, task_gen(TASK, CONFIG)).get()

    setattr(CONFIG, 'activation', nn.ReLU)
    setattr(CONFIG, 'policy_opt_params', dict(lr=3e-4, eps=1e-5))
    setattr(CONFIG, 'value_opt_params', dict(lr=3e-4, eps=1e-5))
    p.map_async(run_bench, task_gen(TASK, CONFIG)).get()

    setattr(CONFIG, 'activation', nn.ReLU)
    setattr(CONFIG, 'policy_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_opt_params', dict(lr=1e-3, eps=1e-5))
    p.map_async(run_bench, task_gen(TASK, CONFIG)).get()

    setattr(CONFIG, 'activation', nn.ReLU)
    setattr(CONFIG, 'policy_opt_params', dict(lr=1e-3))
    setattr(CONFIG, 'value_opt_params', dict(lr=1e-3))
    p.map_async(run_bench, task_gen(TASK, CONFIG)).get()

    setattr(CONFIG, 'activation', nn.Tanh)
    setattr(CONFIG, 'policy_opt_params', dict(lr=1e-3))
    setattr(CONFIG, 'value_opt_params', dict(lr=1e-3))
    p.map_async(run_bench, task_gen(TASK, CONFIG)).get()

    setattr(CONFIG, 'activation', nn.ReLU)
    setattr(CONFIG, 'policy_opt_params', dict(lr=1e-3))
    setattr(CONFIG, 'value_opt_params', dict(lr=1e-3))
    setattr(CONFIG, 'clip_grad_norm', 1.0)
    p.map_async(run_bench, task_gen(TASK, CONFIG)).get()

    setattr(CONFIG, 'activation', nn.Tanh)
    setattr(CONFIG, 'policy_opt_params', dict(lr=1e-3))
    setattr(CONFIG, 'value_opt_params', dict(lr=1e-3))
    setattr(CONFIG, 'clip_grad_norm', 1.0)
    p.map_async(run_bench, task_gen(TASK, CONFIG)).get()
