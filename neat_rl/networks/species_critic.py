import torch
import torch.nn as nn
import torch.nn.functional as F

from neat_rl.networks.util import weights_init_


class SpeciesCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, n_hidden, num_species, emb_dim):
        super().__init__()

        self.species_emb = nn.Embedding(num_species, emb_dim)
        self.in_layer = nn.Sequential(
            nn.Linear(state_dim + action_dim + emb_dim, hidden_size),
            nn.ReLU())

        self.hidden_layers = []
        for _ in range(n_hidden):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        self.out_layer = nn.Linear(hidden_size, 1)

        self.apply(weights_init_)
    
    def forward(self, state, action, species_ids):
        species_embs = self.species_emb(species_ids)
        x = self.in_layer(torch.cat((state, action, species_embs), dim=-1))
        x = self.hidden_layers(x)
        x = self.out_layer(x)
        return x

class SpeciesCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, n_hidden, num_species, emb_size):
        super().__init__()
        self.critic_1 = SpeciesCriticNet(state_dim, action_dim, hidden_size, n_hidden, num_species, emb_size)
        self.critic_2 = SpeciesCriticNet(state_dim, action_dim, hidden_size, n_hidden, num_species, emb_size)

    def Q1(self, state, action, species_ids):
        return self.critic_1(state, action, species_ids)

    def forward(self, state, action, species_ids):
        val_pred_1 = self.critic_1(state, action, species_ids)
        val_pred_2 = self.critic_2(state, action, species_ids)
        
        return val_pred_1, val_pred_2
    
