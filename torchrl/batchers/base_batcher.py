class BaseBatcher:
    '''
    The returned batch will have the shape (num_steps, num_envs, *shape)
    '''

    def __init__(self, env, *, batch_size):
        self.env = env
        self.batch_size = batch_size
        self.batch = None
        self.steps_per_batch = None
        self._state_t = None
        self._state_shape = None

    # TODO: Add get_state shape method
    def get_state_info(self):
        info = self.env.get_state_info()

        if self._state_shape is None:
            state = self.env.reset()
            self._state_shape = self.transform_state(state).shape
        info.shape = self._state_shape[1:]

        return info

    def get_action_info(self):
        return self.env.get_action_info()

    def get_batch(self, select_action_fn):
        if self._state_t is None:
            self._state_t = self.transform_state(self.env.reset())

    def transform_state(self, state):
        '''
        Apply functions to state, called before selecting an action.
        '''
        return state

    def transform_batch(self, batch):
        '''
        Apply functions to batch.
        '''
        return batch
