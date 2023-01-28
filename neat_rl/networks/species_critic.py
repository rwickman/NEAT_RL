import torch
import torch.nn as nn
import torch.nn.functional as F

from neat_rl.networks.util import weights_init_


class SpeciesCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, n_hidden, num_species):
        super().__init__()
        self.device = "cpu"
        self.num_species = num_species
        self.in_layer = nn.Sequential(
            nn.Linear(state_dim + action_dim + num_species, hidden_size),
            nn.ReLU())

        self.hidden_layers = []
        for _ in range(n_hidden):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        self.out_layer = nn.Linear(hidden_size, 1)

        self.apply(weights_init_)
    
    def forward(self, state, action, species_ids):
        species_one_hot = F.one_hot(species_ids.view(-1), self.num_species).to(self.device)
        x = self.in_layer(torch.cat((state, action, species_one_hot), dim=-1))
        x = self.hidden_layers(x)
        x = self.out_layer(x)
        return x

class SpeciesCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, n_hidden, num_species):
        super().__init__()
        self.device = "cpu"
        self.critic_1 = SpeciesCriticNet(state_dim, action_dim, hidden_size, n_hidden, num_species)
        self.critic_2 = SpeciesCriticNet(state_dim, action_dim, hidden_size, n_hidden, num_species)

    def Q1(self, state, action, species_ids):
        return self.critic_1(state, action, species_ids)

    def forward(self, state, action, species_ids):
        val_pred_1 = self.critic_1(state, action, species_ids)
        val_pred_2 = self.critic_2(state, action, species_ids)
        
        return val_pred_1, val_pred_2
    
    def to(self, device):
        self.device = device
        self.critic_1.device = device
        self.critic_2.device = device

        return super().to(device)