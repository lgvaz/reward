import gym, torch
import numpy as np, torch.nn as nn, reward as rw, reward.utils as U
from reward.tfm.img import Gray, Resize, Stack

maxsteps = 40e6
device = U.device.get_device()


class QValueNN(nn.Module):
    def __init__(self, in_channels, n_acs, activation=nn.ReLU):
        super().__init__()
        layers = []
        layers += [nn.Conv2d(in_channels, 32, 8, 4), activation()]
        layers += [nn.Conv2d(32, 64, 4, 2), activation()]
        layers += [nn.Conv2d(64, 64, 3, 1), activation()]
        layers += [rw.nn.Flatten()]
        layers += [nn.Linear(7*7*64, 512), activation()]
        layers += [nn.Linear(512, n_acs)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x): return self.layers(x)

class Policy:
    def __init__(self, qnn, exp_rate):
        self.qnn, self._exp_rate = qnn, U.make_callable(exp_rate)

    @property
    def exp_rate(self): return self._exp_rate(U.global_step.get())

    def get_act(self, s):
        q = self.qnn(s)
        if np.random.random() < self.exp_rate: return U.to_tensor(np.random.choice(len(q.squeeze())), dtype='long', device='cpu')[None]
        else:                                  return q.argmax()[None]


env = U.wrapper.gym.wrap_atari(gym.make('BreakoutNoFrameskip-v4'), clip_rewards=False)
S = rw.space.Image(sz=[1, 84, 84, 4])
A = rw.space.Categorical(n_acs=env.action_space.n)
tfms = [Gray(), Resize(sz=[84, 84]), Stack(n=4)]
exp_rate = U.schedules.linear_schedule(1., .1, int(1e6))
logger = U.Logger('logs/breakout/dqn-unclipped-v3-0', maxsteps=maxsteps)

qnn = QValueNN(in_channels=4, n_acs=env.action_space.n).to(device)
qnn_targ = QValueNN(in_channels=4, n_acs=env.action_space.n).to(device).eval()
q_opt = torch.optim.Adam(qnn.parameters(), lr=1e-4)
policy = Policy(qnn=qnn, exp_rate=exp_rate)
model = rw.model.DQN(policy=policy, qnn=qnn, qnn_targ=qnn_targ, q_opt=q_opt, targ_up_freq=10000, targ_up_w=1., logger=logger)
agent = rw.agent.Replay(model=model, s_sp=S, a_sp=A, bs=32, maxlen=1e6, learn_freq=4, learn_start=5000, logger=logger)

s = env.reset()
r_sum = 0
for i in range(int(maxsteps)):
    s = S(s[None]).apply_tfms(tfms)
    a = agent.get_act(s)
    sn, r, d, _ = env.step(int(a[0].val))
    r_sum += r
    agent.report(r=np.array(np.sign(r))[None], d=np.array(d)[None])
    if d:
        s = env.reset()
        logger.add_log('reward_unclipped', r_sum)
        r_sum = 0
    else: s = sn