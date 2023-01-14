import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from neat_rl.networks.species_actor import SpeciesActor
from neat_rl.networks.species_critic import SpeciesCritic
from neat_rl.networks.discriminator import Discriminator 

from neat_rl.rl.species_replay_buffer import SpeciesReplayBuffer


class SpeciesTD3:
    def __init__(self, args, state_dim, action_dim, max_action):
        self.args = args
        self.max_action = max_action
        self.action_dim = action_dim

        self.args.policy_noise = args.policy_noise * max_action
        self.args.noise_clip = args.noise_clip * max_action
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = SpeciesActor(state_dim, action_dim, self.args.hidden_size, self.args.n_hidden, self.max_action, self.args.max_species, self.args.emb_size).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)

        self.critic = SpeciesCritic(state_dim, action_dim, self.args.hidden_size, self.args.n_hidden, self.args.max_species, self.args.emb_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr)


        self.discriminator = Discriminator(self.args.max_species, state_dim, action_dim, self.args.hidden_size, self.args.n_hidden).to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.disc_lr)

        self.critic_loss_fn = nn.MSELoss()
        self.disc_loss_fn = nn.CrossEntropyLoss()
        self.replay_buffer = SpeciesReplayBuffer(state_dim, action_dim, self.args.replay_capacity)

        self.total_iter = 0

        
    def select_action(self, state, species_id):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state, species_id).cpu().data.numpy().flatten()

    def sample_action(self, state, species_id):
        species_id = torch.tensor([species_id]).to(self.device)

        action = (
            self.select_action(state, species_id)
                + np.random.normal(0, self.max_action * self.args.expl_noise, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
        return action
    
    def train(self):
        self.total_iter += 1

        state, action, next_state, reward, species_id, terminated = self.replay_buffer.sample(self.args.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.args.policy_noise
            ).clamp(-self.args.noise_clip, self.args.noise_clip)

            next_action = (
                self.actor_target(next_state, species_id) + noise
            ).clamp(-self.max_action, self.max_action)
            
            # Add the diversity bonus
            disc_logits = self.discriminator(state, action)
            disc_preds = torch.softmax(disc_logits, dim=-1)
            diversity_bonus = disc_preds.gather(-1, species_id.unsqueeze(1))
            reward = reward + self.args.disc_lam * diversity_bonus 

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action, species_id)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + terminated * self.args.gamma * target_Q
           
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action, species_id)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 2.0)
        self.critic_optimizer.step()

        # Train the discriminator
        logits = self.discriminator(state, action)
        disc_loss = self.disc_loss_fn(logits, species_id)
        self.discriminator_optimizer.zero_grad()
        disc_loss.backward()
        self.discriminator_optimizer.step()

        # Delayed policy updates
        if self.total_iter % self.args.policy_freq == 0:
            #actor_loss = -self.critic.Q1(state, self.actor(state, species_id), species_id).mean()
            actor_loss = self.critic_loss_fn(self.actor(state, species_id), action)
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 2.0)
            self.actor_optimizer.step()

			# Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            if self.total_iter % 512 == 0:
                print("target_Q", target_Q[:10].view(-1), current_Q1[:10].view(-1))
                print("species_id", species_id[:10])
                print("reward", reward[:10].view(-1))
                print("actor_loss", actor_loss, "critic_loss", critic_loss)
                print("diversity_bonus", diversity_bonus.view(-1)[:10], "disc_loss", disc_loss, "\n")

    def save(self):
        model_dict = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict()
        }
        torch.save(model_dict, self.args.rl_save_file)
    
    def load(self):
        model_dict = torch.load(self.args.rl_save_file)

        self.actor.load_state_dict(model_dict["actor"])
        self.actor_target.load_state_dict(model_dict["actor_target"])
        self.actor_optimizer.load_state_dict(model_dict["actor_optimizer"])
        self.critic.load_state_dict(model_dict["critic"])
        self.critic_target.load_state_dict(model_dict["critic_target"])
        self.critic_optimizer.load_state_dict(model_dict["critic_optimizer"])
        self.discriminator.load_state_dict(model_dict["discriminator"])
        self.discriminator_optimizer.load_state_dict(model_dict["discriminator_optimizer"])

