import gym
import torch
import random
# import QDgym
from tqdm import tqdm
import time

from neat_rl.helpers.util import add_to_archive
from neat_rl.rl.species_td3ga import SpeciesTD3GA
from neat_rl.neat.population import GradientPopulation
from neat_rl.helpers.saving import save_population, load_population
from neat_rl.networks.actor import Actor
from neat_rl.networks.sac.sac_models import GaussianPolicy
from neat_rl.rl.species_sac import SpeciesSAC

class EnvironmentGADiversityOrg:
    def __init__(self, args):
        self.args = args

        if self.args.render:
            self.env = gym.make(self.args.env)
            self.env._max_episode_steps = self.args.max_episode_steps
        else:
            self.env = gym.make(self.args.env)
            self.env._max_episode_steps = self.args.max_episode_steps

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0] 
        max_action = float(self.env.action_space.high[0])

        self.args.policy_noise = self.args.policy_noise * max_action
        self.args.noise_clip = self.args.noise_clip * max_action

        if self.args.use_td3_diversity:
            self.td3ga = SpeciesTD3GA(self.args, state_dim, action_dim, max_action, 1)
            base_actor = Actor(state_dim, action_dim, self.args.hidden_size, self.args.n_hidden, max_action)
        else:
            self.td3ga = SpeciesSAC(self.args, state_dim, action_dim, 1, self.env.action_space)
            base_actor = GaussianPolicy(state_dim, action_dim, self.args.hidden_size, self.env.action_space)


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

        species_id = self.population.org_id_to_species[org.id]
        if self.args.render:
            print(species_id)
        cur_step = 0 
        total_diversity_bonus = 0
        total_reward = 0
        # exps = []
        while not done:
            if not self.args.use_td3_diversity:
                if self.td3ga.replay_buffer.size < self.args.learning_starts and not self.args.load and not evaluate:
                    action, log_std = self.env.action_space.sample()
                else:
                    action, log_std, mean = self.td3ga.sample_action_net(org.net, state, evaluate or self.args.render)
                    #action, log_std, mean = self.td3ga.select_action(state, torch.LongTensor([species_id]).to(self.td3ga.device), evaluate or self.args.render)   
            else:
                action, action_org = self.td3ga.sample_action_net(org.net, state, evaluate or self.args.render)

            next_state, reward, done, info = self.env.step(action)
            total_reward += reward

            if not evaluate and not self.args.render:
                if not self.args.use_td3_diversity:
                    self.td3ga.replay_buffer.add(state, action, log_std, mean, next_state, reward, species_id, behavior, done)
                else:
                    self.td3ga.replay_buffer.add(state, action, next_state, reward, species_id, [0], done)

                    
            if not evaluate:
                if self.args.use_state_disc:
                    # inp = torch.cat(
                    #     (torch.FloatTensor(state), torch.FloatTensor(action))).to(self.td3ga.device)
                    total_diversity_bonus += self.td3ga.discriminator(torch.FloatTensor(state).to(self.td3ga.device)).softmax(dim=-1)[species_id].item()
                elif self.args.use_state_only_disc:
                    total_diversity_bonus += self.td3ga.discriminator(torch.FloatTensor(state).to(self.td3ga.device)).softmax(dim=-1)[species_id].item()

                elif self.args.use_action_disc:
                    total_diversity_bonus += self.td3ga.discriminator(torch.FloatTensor(action).to(self.td3ga.device)).softmax(dim=-1)[species_id].item()
                else:
                    disc_logits = self.td3ga.discriminator(torch.FloatTensor(behavior).to(self.td3ga.device))
                    diversity_bonus = disc_logits[species_id].item()

                    total_diversity_bonus += diversity_bonus

            if self.args.render:
                self.env.render()
            if not evaluate and not self.args.render and self.total_timesteps % self.args.update_freq == 0 and self.td3ga.replay_buffer.size >= self.args.batch_size * 8:
                self.td3ga.train()

            state = next_state
            if not evaluate:
                self.total_timesteps += 1
            cur_step += 1
            if self.args.render:
                time.sleep(0.005)

        # if not evaluate and not self.args.render:
        #     for exp in exps:
        #         exp[-2] = behavior
        #         self.td3ga.add_sample(*exp)

        #     self.td3ga.behavior_distr.add(behavior, species_id)
        
        if self.args.render:
            print("total_reward", total_reward, "total_diversity_bonus", total_diversity_bonus)

        return total_reward, total_diversity_bonus
    

                
    def train(self):
        start_time = time.time()
        max_fitness = None
        min_fitness = None
        total_fitness = 0
        random.shuffle(self.population.orgs)

        if self.args.render:
            self.population.orgs = sorted(self.population.orgs, key=lambda x: x.best_fitness, reverse=False)

        for org in self.population.orgs:
            self.total_eval += 1
            total_reward, total_diversity_bonus = self.run(org)
            total_fitness += total_reward
            if self.td3ga.replay_buffer.size >= self.args.learning_starts:
                # Update the organisms behavior
                org.update_fitness(total_reward, total_diversity_bonus)

            if max_fitness is None or total_reward > max_fitness:
                max_fitness = total_reward
            
            if min_fitness is None or total_reward < min_fitness:
                min_fitness = total_reward

        # Train the discriminator
        if not self.args.render:
            self.td3ga.train_discriminator()

        print("Replay buffer size", self.td3ga.replay_buffer.size)
        # if not self.args.render and self.td3ga.replay_buffer.size >= self.args.learning_starts:
        #     self.population.evolve()
        
        avg_fitness = total_fitness / len(self.population.orgs)
        fitness_range = max_fitness - min_fitness
        print("TRAIN TIME: ", time.time() - start_time)

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
        eval_max_fitness = max(fitness_scores)
        return avg_fitness, fitness_scores[0], eval_max_fitness 