import gym, torch
import numpy as np, torch.nn as nn, reward as rw, reward.utils as U
from reward.tfm.img import Gray, Resize, Stack

maxsteps = 40e6
device = U.device.get()

class QValueNN(nn.Module):
    def __init__(self, in_channels, n_acs, dueling=True, activation=nn.ReLU):
        super().__init__()
        self.dueling = dueling
        convs = []
        convs += [nn.Conv2d(in_channels, 32, 8, 4), activation()]
        convs += [nn.Conv2d(32, 64, 4, 2), activation()]
        self.conv_final = nn.Conv2d(64, 64, 3, 1)
        convs += [self.conv_final, activation()]
        convs += [rw.nn.Flatten()]
        self.convs = nn.Sequential(*convs)
        if self.dueling:
            self.v = nn.Sequential(nn.Linear(7*7*64, 512), activation(), nn.Linear(512, 1))
            self.adv = nn.Sequential(nn.Linear(7*7*64, 512), activation(), nn.Linear(512, n_acs))
            self._duel_grad_hook()
        else:
            self.q = nn.Sequential(nn.Linear(7*7*64, 512), activation(), nn.Linear(512, n_acs))

    def forward(self, x):
        x = self.convs(x)
        if self.dueling:
            v, adv = self.v(x), self.adv(x)
            return v + (adv - adv.mean(dim=-1, keepdim=True))
        else:
            return self.q(x)

    def _duel_grad_hook(self):
        def hook(self, grad_input, grad_output):
            for grad in grad_input: grad.mul_(1./np.sqrt(2))
            for grad in grad_output: grad.mul_(1./np.sqrt(2))
        self.conv_final.register_backward_hook(hook)

class Policy:
    def __init__(self, qnn, exp_rate):
        self.qnn, self._exp_rate = qnn, U.make_callable(exp_rate)

    @property
    def exp_rate(self): return self._exp_rate(U.global_step.get())

    def get_act(self, s):
        if np.random.random() < self.exp_rate: return U.to_tensor(A.sample(), dtype='long', device='cpu')
        else:
            q = self.qnn(s)
            return q.argmax(dim=1)


env = U.wrapper.gym.wrap_atari(gym.make('BreakoutNoFrameskip-v4'), clip_rewards=False)
S = rw.space.Image(shape=[1, 84, 84, 4])
A = rw.space.Categorical(n_acs=env.action_space.n)
tfms = [Gray(), Resize(sz=[84, 84]), Stack(n=4)]
exp_rate = U.scheds.PieceLinear(values=[1., .1, .01], bounds=[int(1e6), int(24e6)])
rw.logger.set_logdir('logs/breakout/dddqn-v5-0')
rw.logger.set_maxsteps(maxsteps)

qnn = QValueNN(in_channels=4, n_acs=env.action_space.n).to(device)
qnn_targ = QValueNN(in_channels=4, n_acs=env.action_space.n).to(device).eval()
U.freeze_weights(qnn_targ)
q_opt = U.OptimWrap(torch.optim.Adam(qnn.parameters(), lr=1e-4, eps=1e-4), clip_grad_norm=10.)
policy = Policy(qnn=qnn, exp_rate=exp_rate)
model = rw.model.DQN(policy=policy, qnn=qnn, qnn_targ=qnn_targ, q_opt=q_opt, double=True, targ_up_freq=10000, targ_up_w=1.)
agent = rw.agent.Replay(model=model, s_sp=S, a_sp=A, bs=32, maxlen=1e6, learn_freq=4, learn_start=50000)

s = env.reset()
r_sum = 0
for i in range(int(maxsteps)):
    s = S(s[None]).apply_tfms(tfms)
    a = agent.get_act(s)
    sn, r, d, _ = env.step(int(a[0].val))
    r_sum += r
    agent.report(r=np.array(np.sign(r))[None], d=np.array(d)[None])
    if d:
        if env.env.was_real_done:
            rw.logger.add_log('reward_unclipped', r_sum, force=True)
            r_sum = 0
        s = env.reset()
    else: s = sn