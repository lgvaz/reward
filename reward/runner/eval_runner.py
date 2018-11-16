import pdb
import reward.utils as U
from reward.runner import SingleRunner


class EvalRunner(SingleRunner):
    def __init__(self, env, ep_maxlen=None, tfms=None):
        super().__init__(env=env, ep_maxlen=ep_maxlen)
        self.tfms = tfms or []

    def transform_state(self, state):
        for t in self.tfms:
            state = t.transform_state(state=state, training=False)
        return state

    def run_n_episodes(self, act_fn, num_ep=1):
        for _ in range(num_ep):
            d = False
            state = self.reset()

            while not d:
                state = U.to_tensor(self.transform_state(state))
                ac = act_fn(state)
                state, r, d, info = self.act(ac)

        return self.rs[-num_ep:]

    def write_logs(self, act_fn, logger):
        self.run_n_episodes(act_fn=act_fn, num_ep=1)
        super().write_logs(logger=logger)
