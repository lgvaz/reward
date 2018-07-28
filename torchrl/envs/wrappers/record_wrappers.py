import os
from torchrl.envs.wrappers import BaseWrapper


class GymRecorder(BaseWrapper):
    def __init__(
        self, env, directory, *, video_callable=None, force=True, resume=False, **kwargs
    ):
        from gym.wrappers import Monitor

        if video_callable is None:
            video_callable = lambda x: True
        directory = os.path.join(directory, "videos")

        env.unwrapped.env = Monitor(
            env.unwrapped.env,
            directory=directory,
            video_callable=video_callable,
            force=force,
            resume=resume,
            **kwargs
        )
        super().__init__(env=env)
