import pdb
import reward.utils as U
from reward.runner import SingleRunner


class EvalRunner(SingleRunner):
    def __init__(self, env, ep_maxlen=None, tfms=None):
        super().__init__(env=env, ep_maxlen=ep_maxlen)
        self.tfms = tfms or []

    def transform_s(self, s):
        for t in self.tfms:
            s = t.transform_s(s=s, training=False)
        return s

    def run_n_episodes(self, act_fn, num_ep=1):
        for _ in range(num_ep):
            d = False
            s = self.reset()

            while not d:
                s = U.to_tensor(self.transform_s(s))
                ac = act_fn(s)
                s, r, d, info = self.act(ac)

        return self.rs[-num_ep:]

    def write_logs(self, act_fn, logger):
        self.run_n_episodes(act_fn=act_fn, num_ep=1)
        super().write_logs(logger=logger)
