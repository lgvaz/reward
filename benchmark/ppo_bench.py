import os

import torch.nn as nn

import torchrl.utils as U
from bench import MUJOCO_SIMPLE_BENCH, MUJOCO_ESSENTIAL_BENCH, task_gen
from torchrl.agents import PGAgent
from torchrl.envs import GymEnv, ParallelEnv
from torchrl.models import PPOClipModel, ValueModel
from torchrl.nn import ActionLinear
from torchrl.utils import Config
from utils import config2str

CONFIG = Config(
    activation=nn.Tanh,
    max_steps=4e6,
    steps_per_batch=2048,
    episodes_per_batch=-1,
    normalize_advantages=True,
    normalize_states=True,
    scale_rewards=True,
    advantage=U.estimators.advantage.GAE(gamma=0.99, gae_lambda=0.95),
    policy_opt_params=dict(lr=1e-3, eps=1e-5),
    value_opt_params=dict(lr=1e-3, eps=1e-5),
    value_batch_size=128,
    value_num_epochs=5,
    clip_grad_norm=0.5)

TASK = MUJOCO_ESSENTIAL_BENCH
NUM_WORKERS = 3


def run_bench(config):
    # CONFIG.update(config)
    log_dir = config2str(config)
    log_dir = 'parallel_ppo' + log_dir
    log_dir = os.path.join('logs', config.env_name, log_dir)

    # Create env
    envs = [
        GymEnv(
            env_name=config.env_name,
            normalize_states=config.normalize_states,
            scale_rewards=config.scale_rewards) for _ in range(16)
    ]
    env = ParallelEnv(envs)

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
    policy_model = PPOClipModel.from_config(
        config=policy_model_config,
        env=env,
        opt_params=config.policy_opt_params,
        clip_grad_norm=config.clip_grad_norm)

    value_model_config = Config(nn_config=value_nn_config)
    value_model = ValueModel.from_config(
        config=value_model_config,
        env=env,
        opt_params=config.value_opt_params,
        batch_size=config.value_batch_size,
        num_epochs=config.value_num_epochs,
        clip_grad_norm=config.clip_grad_norm)

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
    import multiprocessing.pool

    class NoDaemonProcess(multiprocessing.Process):
        # make 'daemon' attribute always return False
        def _get_daemon(self):
            return False

        def _set_daemon(self, value):
            pass

        daemon = property(_get_daemon, _set_daemon)

    # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
    # because the latter is only a wrapper function, not a proper class.
    class NoDaemonProcessPool(multiprocessing.pool.Pool):
        Process = NoDaemonProcess

    p = NoDaemonProcessPool(NUM_WORKERS)

    setattr(CONFIG, 'activation', nn.ReLU)
    setattr(CONFIG, 'policy_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'clip_grad_norm', None)
    p.map_async(run_bench, task_gen(TASK, CONFIG)).get()
    p.join()
    p.close()

    p = NoDaemonProcessPool(NUM_WORKERS)
    setattr(CONFIG, 'activation', nn.Tanh)
    setattr(CONFIG, 'policy_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'clip_grad_norm', None)
    p.map_async(run_bench, task_gen(TASK, CONFIG)).get()
    p.join()
    p.close()

    p = NoDaemonProcessPool(NUM_WORKERS)
    setattr(CONFIG, 'activation', nn.Tanh)
    setattr(CONFIG, 'policy_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_batch_size', 64)
    setattr(CONFIG, 'value_num_epochs', 10)
    p.map_async(run_bench, task_gen(TASK, CONFIG)).get()
    p.join()
    p.close()

    p = NoDaemonProcessPool(NUM_WORKERS)
    setattr(CONFIG, 'activation', nn.ReLU)
    setattr(CONFIG, 'policy_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_batch_size', 64)
    setattr(CONFIG, 'value_num_epochs', 10)
    p.map_async(run_bench, task_gen(TASK, CONFIG)).get()
    p.join()
    p.close()

    setattr(CONFIG, 'activation', nn.Tanh)
    setattr(CONFIG, 'policy_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_batch_size', 128)
    setattr(CONFIG, 'value_num_epochs', 5)
    setattr(CONFIG, 'steps_per_batch', 4096)
    p.map_async(run_bench, task_gen(TASK, CONFIG)).get()

    setattr(CONFIG, 'activation', nn.ReLU)
    setattr(CONFIG, 'policy_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_batch_size', 128)
    setattr(CONFIG, 'value_num_epochs', 5)
    setattr(CONFIG, 'steps_per_batch', 4096)
    p.map_async(run_bench, task_gen(TASK, CONFIG)).get()

    setattr(CONFIG, 'activation', nn.Tanh)
    setattr(CONFIG, 'policy_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_batch_size', 128)
    setattr(CONFIG, 'value_num_epochs', 5)
    setattr(CONFIG, 'clip_grad_norm', 0.5)
    p.map_async(run_bench, task_gen(TASK, CONFIG)).get()

    setattr(CONFIG, 'activation', nn.ReLU)
    setattr(CONFIG, 'policy_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_batch_size', 128)
    setattr(CONFIG, 'value_num_epochs', 5)
    setattr(CONFIG, 'clip_grad_norm', 0.5)
    p.map_async(run_bench, task_gen(TASK, CONFIG)).get()

    setattr(CONFIG, 'activation', nn.Tanh)
    setattr(CONFIG, 'policy_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_batch_size', 128)
    setattr(CONFIG, 'value_num_epochs', 5)
    setattr(CONFIG, 'clip_grad_norm', 0.5)
    setattr(CONFIG, 'steps_per_batch', 2048)
    p.map_async(run_bench, task_gen(TASK, CONFIG)).get()

    setattr(CONFIG, 'activation', nn.ReLU)
    setattr(CONFIG, 'policy_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_opt_params', dict(lr=1e-3, eps=1e-5))
    setattr(CONFIG, 'value_batch_size', 128)
    setattr(CONFIG, 'value_num_epochs', 5)
    setattr(CONFIG, 'clip_grad_norm', 0.5)
    setattr(CONFIG, 'steps_per_batch', 2048)
    p.map_async(run_bench, task_gen(TASK, CONFIG)).get()
