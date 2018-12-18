import gym
import reward as rw
import numpy as np
import torch
import torch.nn as nn
import reward.utils as U


MAX_STEPS = 4e5


class QValueNN(nn.Module):
    def __init__(self, n_in, n_acs, activation=nn.ReLU):
        super().__init__()
        layers = []
        layers += [nn.Linear(n_in, 8), activation()]
        layers += [nn.Linear(8, 4), activation()]
        layers += [nn.Linear(4, n_acs)]
        self.layers = nn.Sequential(*layers)

    def forward(self, s): return self.layers(s)


class Policy:
    def __init__(self, qnn, exp_rate):
        self.qnn, self._exp_rate = qnn, U.make_callable(exp_rate)

    @property
    def exp_rate(self): return self._exp_rate(U.global_step.get())

    def get_act(self, s):
        q = self.qnn(s)
        if np.random.random() < self.exp_rate: return U.to_tensor(np.random.choice(len(q.squeeze())), dtype='long', device='cpu')[None]
        else:                                  return q.argmax()[None]

env = gym.make('CartPole-v0').env
S = rw.space.Continuous(low=env.observation_space.low, high=env.observation_space.high)
A = rw.space.Categorical(n_acs=env.action_space.n)
exp_rate = U.schedules.linear_schedule(1., .05, int(.3 * MAX_STEPS))

qnn = QValueNN(n_in=S.shape[0], n_acs=A.n_acs).to(U.device.get())
qnn_targ = QValueNN(n_in=S.shape[0], n_acs=A.n_acs).to(U.device.get()).to(U.device.get()).eval()
q_opt = torch.optim.Adam(qnn.parameters(), lr=1e-4)
policy = Policy(qnn=qnn, exp_rate=exp_rate)

logger = U.Logger('logs/cart_pole/terminal_hack-hardtarg-v4-0', maxsteps=MAX_STEPS)
model = rw.model.DQN(policy=policy, qnn=qnn, qnn_targ=qnn_targ, targ_up_freq=10000, targ_up_w=1., q_opt=q_opt, logger=logger)
agent = rw.agent.Replay(model=model, logger=logger, s_sp=S, a_sp=A, bs=512, maxlen=MAX_STEPS)

s = env.reset()
tsteps = 0
for _ in range(int(MAX_STEPS)):
    a = agent.get_act(S(s[None]))
    s, r, d, _ = env.step(int(a[0].val))
    agent.report(r=np.array(r)[None], d=np.array(d)[None].astype('float'))
    if d or tsteps == 1000:
        s = env.reset()
        tsteps = 0
    tsteps += 1
    logger.add_log('scheds/exploration', exp_rate(U.global_step.get()))


