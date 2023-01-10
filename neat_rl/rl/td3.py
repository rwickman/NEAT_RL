import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from neat_rl.networks.actor import Actor
from neat_rl.networks.critic import Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3:
    def __init__(self, args, state_dim, action_dim, max_action):
        self.args = args
        self.max_action = max_action
        self.action_dim = action_dim
        
        self.actor = Actor(state_dim, action_dim, self.args.hidden_size, self.args.n_hidden, self.max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)

        self.critic = Critic(state_dim, action_dim, self.args.hidden_size, self.args.n_hidden).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr)

        self.critic_loss_fn = nn.MSELoss()

        self.total_iter = 0
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def sample_action(self, state):
        action = (
            self.select_action(state)
                + np.random.normal(0, self.max_action * self.args.expl_noise, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
        return action
    
    def train(self, replay_buffer):
        self.total_iter += 1

        state, action, next_state, reward, terminated = replay_buffer.sample(self.args.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.args.policy_noise
            ).clamp(-self.args.noise_clip, self.args.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)
            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + terminated * self.args.gamma * target_Q    
           
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
        self.critic_optimizer.step()

        if self.total_iter % 512 == 0:
            print("target_Q", target_Q[:10].view(-1), current_Q1[:10].view(-1))
        # Delayed policy updates
        if self.total_iter % self.args.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            #print("actor_loss", actor_loss, "critic_loss", critic_loss)

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            # nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
            actor_loss.backward()
            self.actor_optimizer.step()

			# Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)


    def save(self):
        model_dict = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict()
        }
        torch.save(model_dict, self.args.save_file)
    
    def load(self):
        model_dict = torch.load(self.args.save_file)

        self.actor.load_state_dict(model_dict["actor"])
        self.actor_target.load_state_dict(model_dict["actor_target"])
        self.actor_optimizer.load_state_dict(model_dict["actor_optimizer"])
        self.critic.load_state_dict(model_dict["critic"])
        self.critic_target.load_state_dict(model_dict["critic_target"])
        self.critic_optimizer.load_state_dict(model_dict["critic_optimizer"])
