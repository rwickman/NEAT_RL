import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, n_hidden):
        super().__init__()

        self.in_layer = nn.Linear(state_dim + action_dim, hidden_size)

        self.hidden_layers = []
        for _ in range(n_hidden):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        
        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        self.out_layer = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = F.relu(self.in_layer(torch.cat((state, action), dim=-1)))

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        
        val_pred = self.out_layer(x)
        return val_pred
    



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, n_hidden):
        super().__init__()
        self.critic_1 = CriticNet(state_dim, action_dim, hidden_size, n_hidden)
        self.critic_2 = CriticNet(state_dim, action_dim, hidden_size, n_hidden)

    def Q1(self, state, action):
        return self.critic_1(state, action)

    def forward(self, state, action):
        val_pred_1 = self.critic_1(state, action)
        val_pred_2 = self.critic_2(state, action)
        
        return val_pred_1, val_pred_2
    
