import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


from neat_rl.networks.sac.sac_models import GaussianPolicy
from neat_rl.networks.critic import Critic
from neat_rl.rl.replay_buffer import ReplayBuffer

# TODO: args.sac_alpha

class SAC:
    def __init__(self, args, state_dim, action_dim, action_space=None):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.critic = Critic(state_dim, action_dim, self.args.hidden_size, self.args.n_hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr)
        
        self.actor = GaussianPolicy(state_dim, action_dim, self.args.hidden_size, action_space=action_space).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, self.args.replay_capacity)
        self.total_iter = 0

    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]
    
    def train(self):
        self.total_iter += 1
        state, action, next_state, reward, terminated = self.replay_buffer.sample(self.args.batch_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_state_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.args.sac_alpha * next_state_log_pi
            target_Q = reward + terminated * self.args.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_norm)
        self.critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(state)

        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.args.sac_alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_norm)
        self.actor_optimizer.step()

        if self.total_iter % self.args.policy_freq == 0:
			# Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)


