from reward.utils import Config


def task_gen(tasks, config):
    for task in tasks:
        trials = task.get("trials")

        for i in range(trials):
            new_config = Config(**task)
            new_config.update(config)
            new_config.trial = i
            yield new_config


mujoco_essential_envs = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2"]

mujoco_simple_envs = [
    "Hopper-v2",
    "HalfCheetah-v2",
    "InvertedPendulum-v2",
    "InvertedDoublePendulum-v2",
    "Walker2d-v2",
    "Reacher-v2",
    "Swimmer-v2",
]

roboschool_simple_envs = [
    "RoboschoolHopper-v1",
    "RoboschoolHalfCheetah-v1",
    "RoboschoolInvertedPendulum-v1",
    "RoboschoolInvertedDoublePendulum-v1",
    "RoboschoolWalker2d-v1",
    "RoboschoolReacher-v1",
]

MUJOCO_ESSENTIAL_BENCH = [
    dict(env_name=en, trials=3, max_steps=1e6, steps_per_batch=2048)
    for en in mujoco_essential_envs
]

MUJOCO_SIMPLE_BENCH = [
    dict(env_name=en, trials=4, max_steps=1e6, steps_per_batch=2048)
    for en in mujoco_simple_envs
]

ROBOSCHOOL_SIMPLE_BENCH = [
    dict(env_name=en, trials=4, max_steps=1e6, steps_per_batch=2048)
    for en in roboschool_simple_envs
]
