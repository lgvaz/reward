import gym, torch
import torch.multiprocessing as mp
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
        conv_final = nn.Conv2d(64, 64, 3, 1)
        convs += [conv_final, activation()]
        convs += [rw.nn.Flatten()]
        self.convs = nn.Sequential(*convs)
        if self.dueling:
            self.v = nn.Sequential(nn.Linear(7*7*64, 512), activation(), nn.Linear(512, 1))
            self.adv = nn.Sequential(nn.Linear(7*7*64, 512), activation(), nn.Linear(512, n_acs))
            conv_final.register_backward_hook(self._hook)
        else:
            self.q = nn.Sequential(nn.Linear(7*7*64, 512), activation(), nn.Linear(512, n_acs))

    def forward(self, x):
        x = self.convs(x)
        if self.dueling:
            v, adv = self.v(x), self.adv(x)
            return v + (adv - adv.mean(dim=-1, keepdim=True))
        else:
            return self.q(x)

    @staticmethod
    def _hook(self, grad_input, grad_output):
        for grad in grad_input: grad.mul_(1./np.sqrt(2))
        for grad in grad_output: grad.mul_(1./np.sqrt(2))

class Policy:
    def __init__(self, qnn, exp_rate, a_sp):
        self.qnn, self._exp_rate, self.a_sp = qnn, U.make_callable(exp_rate), a_sp

    @property
    def exp_rate(self): return self._exp_rate(U.global_step.get())

    def get_act(self, s):
        if np.random.random() < self.exp_rate: return self.a_sp.sample()
        else:
            q = self.qnn(s)
            return U.to_np(q.argmax(dim=1))


env_fn = lambda: U.wrapper.gym.wrap_atari(gym.make('PongNoFrameskip-v4'), clip_rewards=False)
env = env_fn()
S = rw.space.Image(shape=[1, 84, 84, 4])
A = rw.space.Categorical(n_acs=env.action_space.n)
tfms = [Gray(), Resize(sz=[84, 84]), Stack(n=4)]
exp_rate = U.scheds.PieceLinear(values=[1., .1, .01], bounds=[int(1e6), int(24e6)])
rw.logger.set_logdir('logs/pong/breakout/dddqn-parallel-targ1k-500kbuffer-v6-2')
rw.logger.set_maxsteps(maxsteps)

qnn = QValueNN(in_channels=4, n_acs=env.action_space.n).to(device)
qnn.share_memory()
qnn_targ = QValueNN(in_channels=4, n_acs=env.action_space.n).to(device).eval()
U.freeze_weights(qnn_targ)
q_opt = U.OptimWrap(torch.optim.Adam(qnn.parameters(), lr=1e-4, eps=1e-4), clip_grad_norm=10.)
policy = Policy(qnn=qnn, exp_rate=exp_rate, a_sp=A)
model = rw.model.DQN(policy=policy, qnn=qnn, qnn_targ=qnn_targ, q_opt=q_opt, double=True, targ_up_freq=1000, targ_up_w=1.)
agent = rw.agent.Replay(model=model, s_sp=S, a_sp=A, bs=32, maxlen=5e5, learn_freq=4, learn_start=50000)
runner = rw.runner.PAAC(env_fn=env_fn, n_envs=8, n_workers=4, s_sp=S, a_sp=A, tfms=tfms)

s = runner.reset()
r_sum = 0
for i in range(int(maxsteps)):
    s = S(s)
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