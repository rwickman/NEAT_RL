import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from neat_rl.networks.util import weights_init_
import copy

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class FakeActionSpace:
    def __init__(self, high, low):
        self.high = high
        self.low = low

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_space=None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.device = device
        return super().to(device)
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.sample(state)
        else:
            _, _, action = self.sample(state)

        return action.detach().cpu().numpy()[0]

    def copy(self, transfer_weights=False):
        copy_net = GaussianPolicy(
            self.state_dim, self.action_dim, self.hidden_dim, FakeActionSpace(self.action_high, self.action_low)).to(self.device)

        if transfer_weights:
            copy_net.load_state_dict(copy.deepcopy(self.state_dict()))
        
        return copy_net

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, log_std



    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
