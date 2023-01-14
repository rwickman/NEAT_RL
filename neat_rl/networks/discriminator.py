import torch
import torch.nn as nn
from neat_rl.networks.util import weights_init_

class Discriminator(nn.Module):
    def __init__(self, num_species, state_dim, action_dim, hidden_size, n_hidden):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_species))

        self.apply(weights_init_)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        logits = self.layers(x)
        return logits

            
        
        
