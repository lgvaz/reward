import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

import torchrl.utils as U


# TODO: Docstring
class BaseAgent(ABC):
    """
    Basic TorchRL agent. Encapsulate an environment and a model.

    Parameters
    ----------
    env: torchrl.envs
        A torchrl environment.
    gamma: float
        Discount factor on future rewards (Default is 0.99).
    log_dir: string
        Directory where logs will be written (Default is `runs`).
    """

    def __init__(self, batcher, optimizer, *, gamma=0.99, log_dir="runs"):
        self.batcher = batcher
        self.opt = optimizer
        self.logger = U.Logger(log_dir)
        self.gamma = gamma
        self.num_iters = 1

        self.models = U.memories.DefaultMemory()
        # Can be changed later by the user, None goes to the default (from policy)
        self.select_action_fn = None
        self.eval_select_action_fn = None

    @abstractmethod
    def step(self):
        """
        This method is called at each interaction of the training loop,
        and defines the training procedure.
        """

    @property
    def num_steps(self):
        return self.batcher.num_steps

    @property
    def num_episodes(self):
        return self.batcher.num_episodes

    def _check_termination(self):
        """
        Check if the training loop reached the end.

        Returns
        -------
        bool
        True if done, False otherwise.
        """
        if self.max_steps is not None:
            self.pbar.update(self.num_steps - self.pbar.n)
            if self.num_steps >= self.max_steps:
                return True
        if self.max_episodes is not None:
            self.pbar.update(self.num_episodes - self.pbar.n)
            if self.num_episodes >= self.max_episodes:
                return True
        if self.max_iters is not None:
            self.pbar.update(self.num_iters - self.pbar.n)
            if self.num_iters >= self.max_iters:
                return True

        return False

    def _check_evaluation(self, env):
        if self.eval_freq is not None and self.num_steps >= self.next_eval:
            action_fn = self.eval_select_action_fn or self.models.policy.select_action
            action_fn_pre = lambda state: action_fn(
                model=self.models.policy,
                state=state,
                step=self.num_steps,
                training=False,
            )
            self.batcher.evaluate(
                select_action_fn=action_fn_pre, logger=self.logger, env=env
            )

            self.next_eval += self.eval_freq

    def register_model(self, name, model):
        """
        Save a torchrl model to the internal memory.

        Parameters
        ----------
        name: str
            Desired name for the model.
        model: torchrl.models
            The model to register.
        """
        setattr(self.models, name, model)
        model.attach_logger(self.logger)

    def train_models(self, batch):
        # for model in self.models.values():
        #     model.train(batch_tensor, step=self.num_steps)
        self.opt.learn_from_batch(batch, step=self.num_steps)

    def train(
        self,
        *,
        max_iters=None,
        max_episodes=None,
        max_steps=None,
        log_freq=1,
        eval_env=None,
        eval_freq=None
    ):
        """
        Defines the training loop of the algorithm, calling :meth:`step` at every iteration.

        Parameters
        ----------
        max_updates: int
            Maximum number of gradient updates (Default is None, meaning it doesn't matter).
        max_episodes: int
            Maximum number of episodes (Default is None, meaning it doesn't matter).
        max_steps: int
            Maximum number of steps (Default is None, meaning it doesn't matter).
        """
        if eval_env is None and eval_freq is not None:
            raise ValueError("eval_env must be specified to use eval_freq")
        stops = np.array([max_iters, max_episodes, max_steps])
        num_stops = sum(stops != None)
        if num_stops != 1:
            raise ValueError(
                "Only one stop criteria should be passed, got {}".format(num_stops)
            )
        self.pbar = tqdm(
            total=stops[stops != None][0], dynamic_ncols=True, unit_scale=True
        )

        self.max_iters = max_iters
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.eval_freq = eval_freq
        self.next_eval = 0

        self.logger.set_log_freq(log_freq=log_freq)

        while True:
            self.step()
            self.write_logs()

            self.num_iters += 1

            self._check_evaluation(env=eval_env)
            if self._check_termination():
                break

        self.pbar.close()

    def select_action(self, state, step, training=True):
        """
        Receive a state and use the model to select an action.

        Parameters
        ----------
        state: numpy.ndarray
            The environment state.

        Returns
        -------
        action: int or numpy.ndarray
            The selected action.
        """
        action_fn = self.select_action_fn or self.models.policy.select_action
        return action_fn(
            model=self.models.policy, state=state, step=step, training=training
        )

    def write_logs(self):
        """
        Use the logger to write general information about the training process.
        """
        self.batcher.write_logs(logger=self.logger)
        self.opt.write_logs(logger=self.logger)

        # self.logger.log(
        #     "Iter {} | Episode {} | Step {}".format(
        #         self.num_iters, self.num_episodes, self.num_steps
        #     )
        # )
        self.logger.log(
            step=self.num_steps,
            header={
                "Iter": self.num_iters,
                "Episode": self.num_episodes,
                "Step": self.num_steps,
            },
        )

    def generate_batch(self):
        batch = self.batcher.get_batch(select_action_fn=self.select_action)
        return batch

    # TODO: Reimplement method
    # @classmethod
    # def from_config(cls, config, env=None):
    #     '''
    #     Create an agent from a configuration object.

    #     Returns
    #     -------
    #     torchrl.agents
    #         A TorchRL agent.
    #     '''
    #     if env is None:
    #         try:
    #             env = U.get_obj(config.env.obj)
    #         except AttributeError:
    #             raise ValueError('The env must be defined in the config '
    #                              'or passed as an argument')

    #     model = cls._model.from_config(config.model, env.get_state_info(), env.get_action_info())

    #     return cls(env, model, **config.agent.as_dict())

    # # TODO: Reimplement method
    # @classmethod
    # def from_file(cls, file_path, env=None):
    #     '''
    #     Create an agent from a configuration file.

    #     Parameters
    #     ----------
    #     file_path: str
    #         Path to the configuration file.

    #     Returns
    #     -------
    #     torchrl.agents
    #         A TorchRL agent.
    #     '''
    #     config = U.Config.load(file_path)

    #     return cls.from_config(config, env=env)
