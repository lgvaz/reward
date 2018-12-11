import gym
import torch
import reward as rw
import numpy as np
import torch.nn as nn
import reward.utils as U

U.device.set_device('cpu')
DEVICE = U.device.get_device()

class PolicyNN(nn.Module):
    def __init__(self, n_ins, n_outs, activation=nn.Tanh):
        super().__init__()
        self.activation = activation()        
        self.hidden = nn.Linear(n_ins, 64)
        self.out = nn.Linear(64, n_outs)
        
    def forward(self, x):       
        return self.out(self.activation(self.hidden(x)))

class Policy:
    def __init__(self, nn):
        self.nn = nn
        
    def get_dist(self, x): return rw.dist.Categorical(logits=self.nn(x))
        
    def get_act(self, x): return self.get_dist(x=x).sample()

    def logprob(self, dist, acs): return dist.log_prob(acs).sum(-1, keepdim=True)
    

# Create environment
env = gym.make('CartPole-v0')
# Define spaces
S = rw.space.Continuous(low=env.observation_space.low, high=env.observation_space.high)
A = rw.space.Categorical(n_acs=env.action_space.n)

pnn = PolicyNN(n_ins=S.shape[0], n_outs=A.n_acs).to(DEVICE)
policy = Policy(nn=pnn)
p_opt = torch.optim.Adam(policy.nn.parameters())
logger = U.Logger('/tmp/tests')
model = rw.model.PG(policy=policy, logger=logger, p_opt=p_opt)
agent = rw.agent.Rollout(model=model, logger=logger, s_sp=S, a_sp=A, bs=512)

s = env.reset()
for _ in range(int(1e5)):
    a = agent.get_act(S(s[None]))
    sn, r, d, _ = env.step(int(a[0].val))
    agent.report(r=np.array(r)[None], d=np.array(d)[None].astype('float'))
    s = sn
    if d: s = env.reset()
