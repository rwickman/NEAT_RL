import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

class Actor(nn.Module):
    def __init__(self, inp_size, out_size, hidden_size, n_hidden, max_action):
        super().__init__()
        self.inp_size = inp_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.max_action = max_action 

        self.in_layer = nn.Linear(inp_size, hidden_size)
        self.hidden_layers = []
        for _ in range(n_hidden):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        
        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        self.out_layer = nn.Linear(hidden_size, out_size)

    def copy(self):
        copy_net = Net(self.inp_size, self.out_size, self.hidden_size, self.n_hidden)
        copy_net.load_state_dict(self.state_dict())

        return copy_net

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        
        x = torch.tanh(self.out_layer(x)) * self.max_action
        return x
    
