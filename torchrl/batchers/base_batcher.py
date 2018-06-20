import torchrl.utils as U


class BaseBatcher:
    '''
    The returned batch will have the shape (num_steps, num_envs, *shape)
    '''

    def __init__(self, runner, *, batch_size):
        self.runner = runner
        self.batch_size = batch_size
        self.batch = None
        self.steps_per_batch = None
        self._state_t = None
        self._state_shape = None

    def __str__(self):
        return '<{}>'.format(type(self).__name__)

    @property
    def unwrapped(self):
        return self

    def get_batch(self, select_action_fn):
        if self._state_t is None:
            self._state_t = self.transform_state(self.runner.reset())

    def transform_state(self, state):
        '''
        Apply functions to state, called before selecting an action.
        '''
        state = U.to_tensor(state)
        return state

    def transform_batch(self, batch):
        '''
        Apply functions to batch.
        '''
        return batch

    # TODO: Add get_state shape method
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
