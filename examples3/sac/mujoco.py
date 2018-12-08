import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import reward as rw
import reward.utils as U

U.device.set_device('cuda')
DEVICE = U.device.get_device()


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


class ValueNN(nn.Module):
    def __init__(self, n_in, hidden=256, activation=nn.ReLU):
        super().__init__()
        layers = []
        layers += [nn.Linear(n_in, hidden), activation()]
        layers += [nn.Linear(hidden, hidden), activation()]
        final_layer = nn.Linear(hidden, 1)
        final_layer.weight.data.uniform_(-3e-3, 3e-3)
        final_layer.bias.data.uniform_(-3e-3, 3e-3)
        layers += [final_layer]
        self.layers = nn.Sequential(*layers)

    def forward(self, x): return self.layers(x)


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

    def logprob(self, dist, acs): return dist.log_prob(acs).sum(-1, keepdim=True)
    def mean(self, dist): return dist.loc
    def std(self, dist): return dist.scale

# ## TODO: Action bounds
env = gym.make('InvertedPendulum-v2')
# Define spaces
S = rw.space.Continuous(low=env.observation_space.low, high=env.observation_space.high)
A = rw.space.Continuous(low=env.action_space.low, high=env.action_space.high)
a_map = U.map_range(-1, 1, A.low[0], A.high[0])

pnn = PolicyNN(n_in=S.shape[0], n_out=A.shape[0]).to(DEVICE)
q1nn = QValueNN(n_in=S.shape[0], n_acs=A.shape[0]).to(DEVICE)
q2nn = QValueNN(n_in=S.shape[0], n_acs=A.shape[0]).to(DEVICE)
vnn = ValueNN(n_in=S.shape[0]).to(DEVICE)
vnn_targ = ValueNN(n_in=S.shape[0]).to(DEVICE).eval()
policy = Policy(nn=pnn)

p_opt = torch.optim.Adam(pnn.parameters())
q1_opt = torch.optim.Adam(q1nn.parameters())
q2_opt = torch.optim.Adam(q2nn.parameters())
v_opt = torch.optim.Adam(vnn.parameters())

model = rw.model.SAC(policy=policy, q1nn=q1nn, q2nn=q2nn, vnn=vnn, vnn_targ=vnn_targ, p_opt=p_opt, q1_opt=q1_opt, q2_opt=q2_opt, v_opt=v_opt, r_scale=1., gamma=0.99)
agent = rw.agent.Replay(model=model, s_sp=S, a_sp=A, bs=256, maxlen=1e6)

s = env.reset()
r_sum = 0


for _ in range(int(1e5)):
    a = agent.get_act(S(s[None]))
    sn, r, d, _ = env.step(a_map(a[0].arr))
    agent.report(r=np.array(r)[None], d=np.array(d)[None].astype('float'))

    s = sn
    r_sum += 1
    if d:
        s = env.reset()
        print('Reward:' + str(r_sum))        
        r_sum = 0

