import gym
import torch
import random

from neat_rl.rl.species_td3ga import SpeciesTD3GA
from neat_rl.neat.population import GradientPopulation
from neat_rl.helpers.saving import save_population, load_population
from neat_rl.networks.actor import Actor

class EnvironmentGADiversity:
    def __init__(self, args, num_episodes=5000):
        self.args = args
        self.num_episodes=num_episodes
        self.env = gym.make(self.args.env)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0] 
        max_action = float(self.env.action_space.high[0])
        self.args.policy_noise = self.args.policy_noise * max_action
        self.args.noise_clip = self.args.noise_clip * max_action

        self.td3ga = SpeciesTD3GA(self.args, state_dim, action_dim, max_action)
        base_actor = Actor(state_dim, action_dim, self.args.hidden_size, self.args.n_hidden, max_action)
        self.total_timesteps = 0
        if self.args.load:
            self.td3ga.load()
            self.population = load_population(self.args, self.td3ga, base_actor)
        else:
            self.population = GradientPopulation(self.args, self.td3ga)
            self.population.setup(base_actor)
        if self.args.render:
            self.env = gym.make(self.args.env, render_mode='human')
        else:
            self.env = gym.make(self.args.env)



    def run(self, org):
        
        state, _ = self.env.reset()
        done = False
        truncated = False
        total_reward = 0
        cur_step = 0
        species_id = self.population.org_id_to_species[org.id]
        if self.args.render:
            print(species_id)
        while not done and not truncated:
            cur_step += 1
            self.total_timesteps += 1
            
            if self.td3ga.replay_buffer.size < self.args.learning_starts and not self.args.load:
                action = self.env.action_space.sample()
            else:
                if self.args.render:
                    action = self.td3ga.get_action_net(org.net, state)
                    #action = self.td3ga.select_action(state, torch.tensor([species_id]).to("cuda"))
                else:
                    action = self.td3ga.sample_action_net(org.net, state)

            next_state, reward, done, truncated, info = self.env.step(action)

            self.td3ga.replay_buffer.add(state, action, next_state, reward, species_id, done)
            if not self.args.render and self.total_timesteps % 32 == 0 and self.td3ga.replay_buffer.size >= self.args.learning_starts:
                self.td3ga.train()
            
            # if self.args.render:
            #     self.env.render()     

            total_reward += reward
            state = next_state
        if self.args.render:
            print(total_reward)
        return total_reward, cur_step

    def train(self):
        max_fitness = None
        best_org = None
        random.shuffle(self.population.orgs)
        if self.args.render:
            self.population.orgs = sorted(self.population.orgs, key=lambda x: x.best_fitness, reverse=True)
        for org in self.population.orgs:
            total_reward, timesteps = self.run(org)

            if self.td3ga.replay_buffer.size >= self.args.learning_starts:
                org.update_fitness(total_reward)

            if max_fitness is None or total_reward > max_fitness:
                max_fitness = total_reward
                best_org = org

        if not self.args.render and self.td3ga.replay_buffer.size >= self.args.learning_starts:
            self.population.evolve()
         

        
        return max_fitness