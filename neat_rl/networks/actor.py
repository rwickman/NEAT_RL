import torch
import torch.nn as nn
import torch.nn.functional as F
from neat_rl.networks.util import weights_init_

import random
import numpy as np
import copy


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, n_hidden, max_action):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.max_action = max_action 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.in_layer = nn.Linear(state_dim, hidden_size)

        self.hidden_layers = []

        for _ in range(n_hidden):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))


        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        self.out_layer = nn.Linear(hidden_size, action_dim)

        self.apply(weights_init_)

    def copy(self, transfer_weights=False):
        copy_net = Actor(self.state_dim, self.action_dim, self.hidden_size, self.n_hidden, self.max_action).to(self.device)
        if transfer_weights:
            copy_net.load_state_dict(copy.deepcopy(self.state_dict()))

        return copy_net

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.forward(state).cpu().data.numpy().flatten()


    def sample_action(self, state):
        action = (
            self.select_action(state)
                + np.random.normal(0, self.max_action * self.args.expl_noise, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
        return action
    

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        
        x = torch.tanh(self.out_layer(x)) * self.max_action
        return x
    
