import gym
import torch
import random
from tqdm import tqdm
import QDgym

from neat_rl.helpers.util import add_to_archive
from neat_rl.rl.species_td3ga import SpeciesTD3GA
from neat_rl.neat.population import GradientPopulation
from neat_rl.helpers.saving import save_population, load_population
from neat_rl.networks.actor import Actor

class EnvironmentGADiversityOrg:
    def __init__(self, args):
        self.args = args

        if self.args.render:
            self.env = gym.make(self.args.env, render=True, max_episode_steps=3000)
        else:
            self.env = gym.make(self.args.env, max_episode_steps=3000)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0] 
        max_action = float(self.env.action_space.high[0])

        self.args.policy_noise = self.args.policy_noise * max_action
        self.args.noise_clip = self.args.noise_clip * max_action

        self.td3ga = SpeciesTD3GA(self.args, state_dim, action_dim, max_action, 1)
        base_actor = Actor(state_dim, action_dim, self.args.hidden_size, self.args.n_hidden, max_action)

        # Total number of timesteps
        self.total_timesteps = 0

        # Total number of organism evaluations
        self.total_eval = 0

        if self.args.load:
            self.td3ga.load()
            self.population = load_population(self.args, self.td3ga, base_actor)
        else:
            self.population = GradientPopulation(self.args, self.td3ga)
            self.population.setup(base_actor)

    def run(self, org, evaluate=False):        
        state = self.env.reset()
        done = False
        total_reward = 0
        import time
        species_id = self.population.org_id_to_species[org.id]
        if self.args.render:
            print(species_id)
        cur_step = 0 
        total_diversity_bonus = 0
        while not done:
            if self.td3ga.replay_buffer.size < self.args.learning_starts and not self.args.load and not evaluate:
                action = self.env.action_space.sample()
            else:
                if self.args.render or evaluate:
                    action = self.td3ga.get_action_net(org.net, state)
                    #action = self.td3ga.sample_action_net(org.net, state)
                    #action = self.td3ga.get_action_net(org.net, state)
                    #action = self.td3ga.select_action(state, torch.tensor([species_id]).to("cuda"))
                else:
                    if self.args.non_exclusive:
                        action = self.td3ga.sample_action_net(org.net, state)
                    else:
                        action = self.td3ga.get_action_net(org.net, state)
                    #action = self.td3ga.sample_action_net(org.net, state)
            

            next_state, reward, done, info = self.env.step(action)
            

            if not evaluate and not self.args.render:
                self.td3ga.replay_buffer.add(state, action, next_state, reward, species_id, [0], done)
            behavior = [0]
            if self.args.render:
                self.args.render()
            if not evaluate:
                

                if self.args.use_state_disc:
                    inp = torch.cat(
                        (torch.FloatTensor(state), torch.FloatTensor(action))).to(self.td3ga.device)
                    total_diversity_bonus += self.td3ga.discriminator(inp).softmax(dim=-1)[species_id].item()
                elif self.args.use_action_disc:
                    total_diversity_bonus += self.td3ga.discriminator(torch.FloatTensor(action).to(self.td3ga.device)).softmax(dim=-1)[species_id].item()
                else:
                    disc_logits = self.td3ga.discriminator(torch.FloatTensor(behavior).to(self.td3ga.device))
                    
                    diversity_bonus = disc_logits.softmax(dim=-1)[species_id].item()

                    total_diversity_bonus += diversity_bonus

            
            if not evaluate and not self.args.render and self.total_timesteps % self.args.update_freq == 0 and self.td3ga.replay_buffer.size >= self.args.batch_size * 8:
                self.td3ga.train()
            
            
            total_reward += reward
            state = next_state
            if not evaluate:
                self.total_timesteps += 1
            cur_step += 1
            if self.args.render:
                time.sleep(0.005)



        if self.args.render:
            print(behavior, cur_step)
        
        if self.args.render:
            print("total_reward", total_reward, "total_diversity_bonus", total_diversity_bonus)

        return total_reward, behavior, total_diversity_bonus
    
    def train_exclusive(self):
        max_fitness = min_fitness = None
        total_fitness = 0

        random.shuffle(self.population.orgs)
        
        for org in self.population.orgs:
            # Run organism if it has not been ran yet
            if org.age == 0:
                assert org.behavior is None
                self.total_eval += 1
                total_reward, behavior = self.run(org)
                
                if self.td3ga.replay_buffer.size >= self.args.learning_starts:
                    # Update the organisms behavior
                    org.behavior = behavior
                    org.update_fitness(total_reward)

                    # Attempt to add to archive
                    if self.kdt is not None:
                        add_to_archive(org, self.archive, self.kdt)


            if self.td3ga.replay_buffer.size >= self.args.learning_starts:
                total_fitness += org.best_fitness
        
                if max_fitness is None or org.best_fitness > max_fitness:
                    max_fitness = org.best_fitness
                
                if min_fitness is None or org.best_fitness < min_fitness:
                    min_fitness = org.best_fitness



        avg_fitness = total_fitness / len(self.population.orgs)
        fitness_range = max_fitness - min_fitness
        return max_fitness, avg_fitness, fitness_range, total_fitness

                
    def train(self):
        max_fitness = None
        min_fitness = None
        total_fitness = 0
        random.shuffle(self.population.orgs)

        if self.args.render:
            self.population.orgs = sorted(self.population.orgs, key=lambda x: x.best_fitness, reverse=True)
            

        for org in self.population.orgs:
            self.total_eval += 1
            total_reward, behavior, total_diversity_bonus = self.run(org)
            total_fitness += total_reward
            if self.td3ga.replay_buffer.size >= self.args.learning_starts:
                # Update the organisms behavior
                org.behavior = behavior
                org.update_fitness(total_reward, total_diversity_bonus)

            if max_fitness is None or total_reward > max_fitness:
                max_fitness = total_reward
            
            if min_fitness is None or total_reward < min_fitness:
                min_fitness = total_reward
        
        print("Replay buffer size", self.td3ga.replay_buffer.size)
        # if not self.args.render and self.td3ga.replay_buffer.size >= self.args.learning_starts:
        #     self.population.evolve()
        
        avg_fitness = total_fitness / len(self.population.orgs)
        fitness_range = max_fitness - min_fitness

        return max_fitness, avg_fitness, fitness_range, total_fitness

    def evaluate_10(self, org):
        print("Running evaluation")
        fitness_scores = []
        for _ in tqdm(range(10)):
            total_reward, _, _ = self.run(org, evaluate=True)
            fitness_scores.append(total_reward)

        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        #fitness_diff = fitness_scores[0] - avg_fitness
        fitness_diff = abs(fitness_scores[0] - avg_fitness) / (abs(fitness_scores[0] + avg_fitness) / 2)
        print("fitness_diff", fitness_scores, fitness_diff, org.best_fitness)
        return avg_fitness, fitness_scores[0] 
             
