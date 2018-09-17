from abc import ABC, abstractmethod
from tqdm import tqdm
import reward.utils as U


class BaseBatcher(ABC):
    """
    The returned batch will have the shape (num_steps, num_envs, *shape)
    """

    def __init__(self, runner, batch_size, transforms=None):
        self.runner = runner
        self.batch_size = batch_size
        self.transforms = transforms or []
        self.batch = None
        self.state_t = None
        self._state_shape = None

    def __str__(self):
        return "<{}>".format(type(self).__name__)

    @abstractmethod
    def get_batch(self, select_action_fn):
        if self.state_t is None:
            self.state_t = self.transform_state(self.runner.reset())

    @property
    def env_name(self):
        return self.runner.env_name

    @property
    def unwrapped(self):
        return self

    @property
    def num_steps(self):
        return self.runner.num_steps

    @property
    def num_episodes(self):
        return self.runner.num_episodes

    def get_batches(self, max_steps, select_action_fn):
        pbar = tqdm(total=max_steps, dynamic_ncols=True, unit_scale=True)
        while self.num_steps < max_steps:
            yield self.get_batch(select_action_fn=select_action_fn)
            pbar.update(self.num_steps - pbar.n)

        pbar.close()

    def transform_state(self, state, training=True):
        """
        Apply functions to state, called before selecting an action.
        """
        # TODO
        # state = U.to_tensor(state)
        # state = U.to_np(state)
        for t in self.transforms:
            state = t.transform_state(state, training=training)
        return state

    def transform_batch(self, batch, training=True):
        """
        Apply functions to batch.
        """
        for t in self.transforms:
            batch = t.transform_batch(batch, training=training)
        return batch

    def evaluate(self, env, select_action_fn, logger):
        self.runner.evaluate(
            env=env,
            select_action_fn=select_action_fn,
            state_transform=self.transform_state,
            logger=logger,
        )

    # TODO: Now with transforms this is more straight-forward. RE-IMPLEMENT
    def get_state_info(self):
        info = self.runner.get_state_info()

        if self._state_shape is None:
            state = self.runner.reset()
            self._state_shape = tuple(self.transform_state(state).shape)
        info.shape = self._state_shape[1:]

        return info

    def get_action_info(self):
        return self.runner.get_action_info()

    def write_logs(self, logger):
        self.runner.write_logs(logger)

    def close(self):
        self.runner.close()
