import gym
import reward as rw

class PolicyNN(nn.Module):
    def __init__(self, n_ins, n_outs, activation=nn.Tanh):
        super().__init__()
        self.activation = activation()        
        self.hidden = nn.Linear(n_ins, 64)
        self.out = nn.Linear(64, n_outs)
        
    def forward(self, x):       
        return self.out(self.activation(self.hidden(x)))

# Create environment
env = gym.make('CartPole-v0')
# Define spaces
S = rw.space.Continuous(low=env.observation_space.low, high=env.observation_space.high)
A = rw.space.Categorical(n_acs=env.action_space.n)