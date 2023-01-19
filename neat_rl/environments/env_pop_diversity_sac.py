import gym
import torch
import random
import QDgym
from tqdm import tqdm

from neat_rl.helpers.util import add_to_archive
from neat_rl.rl.species_sac import SpeciesSAC
from neat_rl.neat.population import GradientPopulation
from neat_rl.helpers.saving import save_population, load_population
from neat_rl.networks.sac.sac_models import GaussianPolicy

class EnvironmentGADiversitySAC:
    def __init__(self, args, archive, kdt):
        self.args = args
        self.archive = archive
        self.kdt = kdt

        
        self.env = gym.make(self.args.env, render=self.args.render)
        

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0] 

        self.sac = SpeciesSAC(self.args, state_dim, action_dim, len(self.env.desc), self.env.action_space)
        base_actor = GaussianPolicy(state_dim, action_dim, self.args.hidden_size, self.env.action_space)

        self.population = GradientPopulation(self.args, self.sac)
        self.population.setup(base_actor)
        
        self.total_timesteps = 0
        if self.args.load:
            self.sac.load()
            self.population = load_population(self.args, self.sac, base_actor)
        else:
            self.population = GradientPopulation(self.args, self.sac)
            self.population.setup(base_actor)

    def run(self, org, evaluate=False):        
        state = self.env.reset()
        done = False
        total_reward = 0
        total_diversity_bonus = 0
        import time
        species_id = self.population.org_id_to_species[org.id]
        if self.args.render:
            print(species_id)
        cur_step = 0 
        while not done:
            if self.sac.replay_buffer.size < self.args.learning_starts and not self.args.load and not evaluate:
                action = self.env.action_space.sample()
            else:
                action = org.net.select_action(state, evaluate or self.args.render)

            next_state, reward, done, info = self.env.step(action)
            behavior = self.env.desc

            if not evaluate and not self.args.render:
                self.sac.replay_buffer.add(state, action, next_state, reward, species_id, behavior, done)
                total_diversity_bonus += self.sac.discriminator(torch.FloatTensor(behavior).to(self.sac.device))[species_id].item()
            
            if not evaluate and not self.args.render and self.total_timesteps % self.args.update_freq == 0 and self.sac.replay_buffer.size >= self.args.batch_size * 8:
                self.sac.train()
            
            
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
            print(total_reward)

        return total_reward, behavior, total_diversity_bonus


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
            if self.sac.replay_buffer.size >= self.args.learning_starts:
                # Update the organisms behavior
                org.behavior = behavior
                org.update_fitness(total_reward, total_diversity_bonus)

                # Attempt to add to archive
                if self.kdt is not None:
                    add_to_archive(org, self.archive, self.kdt)

            if max_fitness is None or total_reward > max_fitness:
                max_fitness = total_reward
            
            if min_fitness is None or total_reward < min_fitness:
                min_fitness = total_reward
        
        print("Replay buffer size", self.sac.replay_buffer.size)
        # if not self.args.render and self.sac.replay_buffer.size >= self.args.learning_starts:
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
             
