import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os

from neat_rl.networks.sac.species_sac_models import SpeciesGaussianPolicy
from neat_rl.networks.species_critic import SpeciesCritic
from neat_rl.rl.species_replay_buffer import SpeciesReplayBuffer
from neat_rl.networks.discriminator import Discriminator

# TODO: args.sac_alpha

class SpeciesSAC:
    def __init__(self, args, state_dim, action_dim, behavior_dim, action_space=None):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.critic = SpeciesCritic(state_dim, action_dim, self.args.critic_hidden_size, self.args.n_hidden, self.args.num_species, self.args.emb_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr)
        
        self.actor = SpeciesGaussianPolicy(state_dim, action_dim, self.args.critic_hidden_size, self.args.num_species, self.args.emb_size, action_space=action_space).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.replay_buffer = SpeciesReplayBuffer(state_dim, action_dim, behavior_dim, self.args.replay_capacity)
        self.total_iter = 0
        if self.args.use_state_disc:
            self.discriminator = Discriminator(self.args.num_species, state_dim + action_dim, self.args.critic_hidden_size, self.args.n_hidden).to(self.device)
        else:
            self.discriminator = Discriminator(self.args.num_species, behavior_dim, self.args.critic_hidden_size, self.args.n_hidden).to(self.device)

        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.disc_lr)
        self.disc_loss_fn = nn.CrossEntropyLoss()

        self.rl_save_file = os.path.join(self.args.save_dir, "sac.pt")

    def select_action(self, state, species_id, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.actor.sample(state, species_id)
        else:
            _, _, action = self.actor.sample(state, species_id)

        return action.detach().cpu().numpy()[0]
    
    def sample_action_net(self, net, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = net.sample(state)
        else:
            _, _, action = net.sample(state)

        return action.detach().cpu().numpy()[0]

    def pg_update(self, net, species_id):
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.org_lr)
        species_ids = torch.full((self.args.batch_size,), species_id, device=self.device)
        for _ in range(self.args.n_org_updates):
            state  = self.replay_buffer.sample_states(self.args.batch_size)
            
            #torch.IntTensor([species_id]).to(self.device)
            pi, log_pi, _ = net.sample(state)
            qf1_pi, qf2_pi = self.critic(state, pi, species_ids)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((self.args.sac_alpha * log_pi) - min_qf_pi).mean()

            optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), self.args.max_norm)
            optimizer.step()


    def train(self):
        self.total_iter += 1
        state, action, next_state, reward, species_id, behavior, terminated = self.replay_buffer.sample(self.args.batch_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi, mean = self.actor.sample(next_state, species_id)

            # Add the diversity bonus
            if self.args.use_state_disc:
                logits = self.discriminator(torch.cat((state, action), -1))
            else:
                logits = self.discriminator(behavior)

            disc_preds = torch.softmax(logits, dim=-1)
            diversity_bonus = disc_preds.gather(-1, species_id.unsqueeze(1))
            reward = reward + self.args.disc_lam * diversity_bonus 
            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_state_action, species_id)
            target_Q = torch.min(target_Q1, target_Q2) - self.args.sac_alpha * next_state_log_pi
            target_Q = reward + terminated * self.args.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action, species_id)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_norm)
        self.critic_optimizer.step()

        # Train the discriminator
        if self.args.use_state_disc:
            logits = self.discriminator(torch.cat((state, action), -1))
        else:
            logits = self.discriminator(behavior)

        disc_loss = self.disc_loss_fn(logits, species_id)
        self.discriminator_optimizer.zero_grad()
        disc_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.args.max_norm)
        self.discriminator_optimizer.step()




        pi, log_pi, _ = self.actor.sample(state, species_id)
        actor_loss = F.mse_loss(pi, action)

        # qf1_pi, qf2_pi = self.critic(state, pi, species_id)
        # min_qf_pi = torch.min(qf1_pi, qf2_pi)
        # actor_loss = ((self.args.sac_alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_norm)
        self.actor_optimizer.step()
        if self.total_iter % 512 == 0:
            print("target_Q", target_Q[:10].view(-1), current_Q1[:10].view(-1))
            print("species_id", species_id[:10])
            print("reward", reward[:10].view(-1))
            print("actor_loss", actor_loss, "critic_loss", critic_loss)
            print("diversity_bonus", diversity_bonus.view(-1)[:10], "disc_loss", disc_loss, "\n")

        if self.total_iter % self.args.policy_freq == 0:
			# Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)


    def save(self):
        model_dict = {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict()
        }
        torch.save(model_dict, self.rl_save_file)
    
    def load(self):
        model_dict = torch.load(self.rl_save_file)

        self.actor.load_state_dict(model_dict["actor"])
        self.actor_optimizer.load_state_dict(model_dict["actor_optimizer"])
        self.critic.load_state_dict(model_dict["critic"])
        self.critic_target.load_state_dict(model_dict["critic_target"])
        self.critic_optimizer.load_state_dict(model_dict["critic_optimizer"])
        self.discriminator.load_state_dict(model_dict["discriminator"])
        self.discriminator_optimizer.load_state_dict(model_dict["discriminator_optimizer"])

