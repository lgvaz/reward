import gym
import torch
import numpy as np, reward as rw, torch.nn as nn, matplotlib.pyplot as plt
import torch.nn.functional as F, reward.utils as U

screen_width = 600
device = U.device.get_device()
MAX_STEPS = 2e5

def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:                    slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2): slice_range = slice(-view_width, None)
    else: slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    return screen[:, :, slice_range]


class QValueNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class Policy:
    def __init__(self, qnn, exp_rate):
        self.qnn, self._exp_rate = qnn, U.make_callable(exp_rate)

    @property
    def exp_rate(self): return self._exp_rate(U.global_step.get())

    def get_act(self, s):
        q = self.qnn(s)
        if np.random.random() < self.exp_rate: return U.to_tensor(np.random.choice(len(q.squeeze())), dtype='long', device='cpu')[None]
        else:                                  return q.argmax()[None]


env = gym.make('CartPole-v0').unwrapped
def main():
    S = rw.space.Image(sz=[40, 80], order='NCHW')
    A = rw.space.Categorical(n_acs=env.action_space.n)
    exp_rate = U.schedules.linear_schedule(1., .1, int(.3 * MAX_STEPS))
    tfms = [rw.tfm.img.Gray(), rw.tfm.img.Resize(sz=[40, 80]), rw.tfm.img.Stack(n=4)]

    qnn = QValueNN().to(device)
    qnn_targ = QValueNN().to(device).eval()
    q_opt = torch.optim.Adam(qnn.parameters())
    policy = Policy(qnn=qnn, exp_rate=exp_rate)

    logger = U.Logger('/tmp/logs/cp_img/v3-1', maxsteps=MAX_STEPS, logfreq=300)
    model = rw.model.DQN(policy=policy, qnn=qnn, qnn_targ=qnn_targ, q_opt=q_opt, targ_up_freq=10, logger=logger, gamma=0.99)
    agent = rw.agent.Replay(model=model, logger=logger, s_sp=S, a_sp=A, bs=128, maxlen=1e4)

    state = env.reset()
    last_screen = get_screen()
    new_screen = get_screen()
    tsteps = 0

    for _ in range(int(MAX_STEPS)):
        s = S((new_screen - last_screen)[None])
        s = s.apply_tfms(tfms)
        a = agent.get_act(s)
        _, r, d, _ = env.step(int(a[0].val))
        new_screen, last_screen = get_screen(), new_screen
        agent.report(r=np.array(r)[None], d=np.array(d)[None].astype('float'))
        if d or tsteps == 1000:
            _ = env.reset()
            new_screen, last_screen = get_screen(), new_screen
            tsteps = 0
        tsteps += 1
        logger.add_log('scheds/exploration', exp_rate(U.global_step.get()))

if __name__ == '__main__': main()

