import torch, torch.nn as nn, numpy as np
import reward as rw, reward.utils as U
from dm_control import suite, viewer

DEVICE = U.device.get()

class PolicyNN(nn.Module):
    def __init__(self, n_in, n_out, hidden=256, activation=nn.ReLU, logstd_range=(-20, 2)):
        super().__init__()
        self.logstd_range = logstd_range

        layers = []
        layers += [nn.Linear(n_in, hidden), activation()]
        layers += [nn.Linear(hidden, hidden), activation()]
        self.layers = nn.Sequential(*layers)
        self.mean = nn.Linear(hidden, n_out)
        self.mean.weight.data.uniform_(-3e-3, 3e-3)
        self.mean.bias.data.uniform_(-3e-3, 3e-3)
        self.log_std = nn.Linear(hidden, n_out)
        self.log_std.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = self.layers(x)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(*self.logstd_range)
        return mean, log_std

class Policy:
    def __init__(self, nn): self.nn = nn
        
    def get_dist(self, s):
        mean, log_std = self.nn(s)
        return rw.dist.TanhNormal(loc=mean, scale=log_std.exp())

    def get_act(self, s=None, dist=None):
        assert (s is not None and dist is None) or (s is None and dist is not None)
        dist = dist or self.get_dist(s=s)
        return dist.rsample()

    def get_act_pre(self, s=None, dist=None):
        assert (s is not None and dist is None) or (s is None and dist is not None)
        dist = dist or self.get_dist(s=s)
        return dist.rsample_with_pre()

    def logprob(self, dist, acs): return dist.log_prob(acs).sum(-1, keepdim=True)
    def logprob_pre(self, dist, acs): return dist.log_prob_pre(acs).sum(-1, keepdim=True)
    def mean(self, dist): return dist.loc
    def std(self, dist): return dist.scale


def concat_state_shape(s_spec): return (int(np.sum([np.prod(o.shape) for o in s_spec.values()])), )
def concat_state(s): return np.concatenate([o.flatten() for o in s.values()])

def get_act_fn(policy, a_map):
    def get(tstep):
        s = S(concat_state(tstep.observation)[None]).to_tensor()
        return a_map(U.to_np(policy.get_act(s)[0]))
    return get

# env = suite.load(domain_name="cartpole", task_name="three_poles")
env = suite.load(domain_name="walker", task_name="run")
# Define spaces
S = rw.space.Continuous(shape=concat_state_shape(env.observation_spec()), low=-np.inf, high=np.inf)
A = rw.space.Continuous(low=env.action_spec().minimum, high=env.action_spec().maximum, shape=env.action_spec().shape)
a_map = U.map_range(-1, 1, A.low[0], A.high[0])

pnn = PolicyNN(n_in=S.shape[0], n_out=A.shape[0]).to(DEVICE)
policy = Policy(nn=pnn)
U.load_model(pnn, path='logs/dm/walker/run-max999-v9-2/models/pnn_checkpoint')
get_act = get_act_fn(policy=policy, a_map=a_map)

viewer.launch(env, policy=get_act)
