import torch
import torch.nn as nn
from neat_rl.networks.util import weights_init_

class Discriminator(nn.Module):
    def __init__(self, num_species: int, state_dim: int, action_dim: int, hidden_size: int, n_hidden: int):
        super().__init__()

        self.state_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_species))


        self.action_layers = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_species))

        self.apply(weights_init_)

    def forward(self, state, action):
        logits_state = self.state_layers(state)
        logits_action = self.action_layers(action)
        return logits_state, logits_action

            
        
        
