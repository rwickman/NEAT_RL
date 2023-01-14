import torch
import torch.nn as nn

from neat_rl.networks.util import weights_init_

class SpeciesActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, n_hidden, max_action, num_species, emb_dim):
        super().__init__()
        self.max_action = max_action
        self.species_emb = nn.Embedding(num_species, emb_dim)
        self.in_layer = nn.Sequential(
            nn.Linear(state_dim + emb_dim, hidden_size),
            nn.ReLU())

        self.hidden_layers = []
        for _ in range(n_hidden):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        self.out_layer = nn.Linear(hidden_size, action_dim)

        self.apply(weights_init_)
        
    
    def forward(self, x, species_id):
        
        species_embs = self.species_emb(species_id)
        x = self.in_layer(torch.cat((x, species_embs), dim=-1))
        x = self.hidden_layers(x)
        x = torch.tanh(self.out_layer(x)) * self.max_action
        return x
        