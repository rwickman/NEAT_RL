import gym
import torch
import random

from neat_rl.rl.species_sac import SpeciesSAC
from neat_rl.neat.population import GradientPopulation
from neat_rl.helpers.saving import save_population, load_population
from neat_rl.networks.sac.sac_models import GaussianPolicy

from torch.multiprocessing import Queue, Process

def train_run(org, org_idx, species_id, exp_queue, done_queue, env_name, should_sample):
    env = gym.make(env_name)
    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    cur_step = 0

    while not done and not truncated:
        cur_step += 1
        
        if should_sample:
            action = env.action_space.sample()
        else:
            action = org.net.select_action(state)

        next_state, reward, done, truncated, info = env.step(action)    

        exp_queue.put((state, action, next_state, reward, species_id, done))

        total_reward += reward
        state = next_state
    
    done_queue.put((org_idx, total_reward))


class EnvironmentGADiversitySAC:
    def __init__(self, args):
        self.args = args

        if self.args.render:
            self.env = gym.make(self.args.env, render_mode='human')
        else:
            self.env = gym.make(self.args.env)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0] 

        self.sac = SpeciesSAC(self.args, state_dim, action_dim, self.env.action_space)
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
            
            if self.sac.replay_buffer.size < self.args.learning_starts and not self.args.load:
                action = self.env.action_space.sample()
            else:
                action = self.sac.select_action_net(org.net, state, self.args.render)

            next_state, reward, done, truncated, info = self.env.step(action)

            self.sac.replay_buffer.add(state, action, next_state, reward, species_id, done)
            if self.total_timesteps % 32 == 0 and self.sac.replay_buffer.size >= self.args.learning_starts:
                self.sac.train()
            
            if self.args.render:
                self.env.render()     

            total_reward += reward
            state = next_state

        return total_reward, cur_step



    def train(self):
        max_fitness = None
        best_org = None
        random.shuffle(self.population.orgs)
        exp_queue = Queue()
        done_queue = Queue()
        #self.population.orgs = sorted(self.population.orgs, key=lambda x: x.best_fitness, reverse=True)

        # Spawn the processes
        should_sample = not self.args.load and self.sac.replay_buffer.size < self.args.learning_starts
        for org_idx, org in enumerate(self.population.orgs):
            species_id = self.population.org_id_to_species[org.id]
            org.net.to("cpu")
            
            process = Process(target = train_run, args=(
                org, org_idx, species_id, exp_queue, done_queue, self.args.env, should_sample))
            process.start()
        
        num_done = 0
        while num_done < len(self.population.orgs):
            if not done_queue.empty():
                num_done += 1
                org_idx, total_reward = done_queue.get()
                
                org = self.population.orgs[org_idx]
                org.net.to(self.sac.device)
                
                if max_fitness is None or total_reward > max_fitness:
                    max_fitness = total_reward
                    best_org = org

                if self.sac.replay_buffer.size >= self.args.learning_starts:
                    org.update_fitness(total_reward)
            
            while not exp_queue.empty():
                self.sac.replay_buffer.add(*exp_queue.get())
                self.total_timesteps += 1
                if self.total_timesteps % 32 == 0 and self.sac.replay_buffer.size >= self.args.learning_starts:
                    self.sac.train()
                    
            


        if self.sac.replay_buffer.size >= self.args.learning_starts:
            self.population.evolve()
         

        
        return max_fitness





    def test(self):
        max_fitness = None
        best_org = None
        random.shuffle(self.population.orgs)

        #self.population.orgs = sorted(self.population.orgs, key=lambda x: x.best_fitness, reverse=True)
        for org in self.population.orgs:
            total_reward, timesteps = self.run(org)

            if self.sac.replay_buffer.size >= self.args.learning_starts:
                org.update_fitness(total_reward)

            if max_fitness is None or total_reward > max_fitness:
                max_fitness = total_reward
                best_org = org

        if self.sac.replay_buffer.size >= self.args.learning_starts:
            self.population.evolve()
         

        
        return max_fitness
