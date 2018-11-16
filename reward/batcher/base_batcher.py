import reward.utils as U
from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm
from boltons.cacheutils import cachedproperty


class BaseBatcher(ABC):
    """
    The returned batch will have the shape (num_steps, num_envs, *shape)
    """

    def __init__(self, runner, batch_size, transforms=None):
        self.runner = runner
        self.batch_size = batch_size
        self.transforms = transforms or []
        self.batch = None
        self.s = None

    def __str__(self):
        return "<{}>".format(type(self).__name__)

    @abstractmethod
    def get_batch(self, act_fn):
        pass

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

    @cachedproperty
    def state_space(self):
        space = self.runner.state_space
        state = self.runner.reset()
        space.shape = tuple(self.transform_state(state).shape)[1:]
        return space

    @property
    def ac_space(self):
        return self.runner.ac_space

    @property
    def is_best(self):
        return self.runner.is_best

    def get_batches(self, max_steps, act_fn):
        pbar = tqdm(total=max_steps, dynamic_ncols=True, unit_scale=True)
        while self.num_steps < max_steps:
            yield self.get_batch(act_fn=act_fn)
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

    def write_logs(self, logger):
        self.runner.write_logs(logger)

    def close(self):
        self.runner.close()
