import torch
import torch.nn as nn
from neat_rl.networks.util import weights_init_

class Discriminator(nn.Module):
    def __init__(self, num_species: int, behavior_dim: int, hidden_size: int, n_hidden: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(behavior_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_species))

        self.apply(weights_init_)

    def forward(self, behavior):
        logits = self.layers(behavior)
        return logits

            
        
        
