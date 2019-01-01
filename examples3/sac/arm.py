import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import reward as rw
import reward.utils as U

U.device.set_device('cuda')
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

class QValueNN(nn.Module):
    def __init__(self, n_in, n_acs, hidden=256, activation=nn.ReLU):
        super().__init__()
        layers = []
        layers += [nn.Linear(n_in + n_acs, hidden), activation()]
        layers += [nn.Linear(hidden, hidden), activation()]
        final_layer = nn.Linear(hidden, 1)
        final_layer.weight.data.uniform_(-3e-3, 3e-3)
        final_layer.bias.data.uniform_(-3e-3, 3e-3)
        layers += [final_layer]
        self.layers = nn.Sequential(*layers)

    def forward(self, s, a): return self.layers(torch.cat([s, a], dim=1))

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

def encode_state(state):
    goal_diff = state['desired_goal'] - state['desired_goal']
    return np.concatenate((goal_diff, state['observation']))

env = gym.make('FetchPush-v1')
env.env.reward_type = 'dense'
# Define spaces
n_in = env.observation_space.spaces['achieved_goal'].shape[0] + env.observation_space.spaces['observation'].shape[0]
S = rw.space.Continuous(low=-np.inf, high=np.inf, shape=(n_in,))
A = rw.space.Continuous(low=env.action_space.low, high=env.action_space.high)
a_map = U.map_range(-1, 1, A.low[0], A.high[0])

pnn = PolicyNN(n_in=S.shape[0], n_out=A.shape[0]).to(DEVICE)
q1nn = QValueNN(n_in=S.shape[0], n_acs=A.shape[0]).to(DEVICE)
q2nn = QValueNN(n_in=S.shape[0], n_acs=A.shape[0]).to(DEVICE)
policy = Policy(nn=pnn)

p_opt = torch.optim.Adam(pnn.parameters(), lr=1e-3)
q1_opt = torch.optim.Adam(q1nn.parameters(), lr=1e-3)
q2_opt = torch.optim.Adam(q2nn.parameters(), lr=1e-3)

rw.logger.set_logdir('logs/fetch2/push/v9_lr1e3-0')
rw.logger.set_maxsteps(20e6)
entropy = -np.prod(env.action_space.shape)
model = rw.model.SAC(policy=policy, q1nn=q1nn, q2nn=q2nn, p_opt=p_opt, q1_opt=q1_opt, q2_opt=q2_opt, entropy=entropy)
agent = rw.agent.Replay(model=model, s_sp=S, a_sp=A, bs=256, maxlen=1e6)

s = env.reset()
n_goals, n_success = 1, 0
for i in range(int(20e6)):
    a = agent.get_act(S(encode_state(s)[None]))
    s, r, d, info = env.step(a_map(a[0].arr[0]))
    agent.report(r=np.array(r)[None], d=np.array(d)[None])
    if info['is_success']: 
        n_goals += 1
        n_success += 1
    if d:
        n_goals += 1
        s = env.reset()
    if (i+1) % 1000 == 0:
        rw.logger.add_log('episode/success_rate', n_success / n_goals)
        n_goals, n_success = 1, 0