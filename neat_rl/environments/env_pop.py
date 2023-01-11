import gym
import torch
from neat_rl.rl.td3ga import TD3GA
from neat_rl.neat.population import GradientPopulation
from neat_rl.helpers.saving import save_population, load_population
class EnvironmentGA:
    def __init__(self, args, num_episodes=5000):
        self.args = args
        self.num_episodes=num_episodes
        self.env = gym.make(self.args.env)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0] 
        max_action = float(self.env.action_space.high[0])
        self.args.policy_noise = self.args.policy_noise * max_action
        self.args.noise_clip = self.args.noise_clip * max_action

        self.td3ga = TD3GA(self.args, state_dim, action_dim, max_action)
        self.population = GradientPopulation(self.args, self.td3ga)
        
        self.population.setup(self.td3ga.actor)

        if self.args.load:
            self.td3ga.load()
            self.population = load_population(self.args, self.td3ga)

    def run(self, org, render=False):
        if render:
            self.env = gym.make(self.args.env, render_mode='human')
        else:
            self.env = gym.make(self.args.env)


        state, _ = self.env.reset()
        done = False
        truncated = False
        total_reward = 0
        cur_step = 0

        while not done and not truncated:
            cur_step += 1
            if self.td3ga.replay_buffer.size < self.args.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.td3ga.sample_action_net(org.net, state)

            next_state, reward, done, truncated, info = self.env.step(action)    
            if cur_step > self.args.max_timesteps:
                done = True

            self.td3ga.replay_buffer.add(state, action, next_state, reward, done)
            if self.td3ga.replay_buffer.size >= self.args.learning_starts:
                self.td3ga.train()
            
            if render:
                self.env.render()     

            total_reward += reward
            state = next_state

        return total_reward, cur_step

    def train(self):
        max_fitness = None
        best_org = None
        for org in self.population.orgs:
            total_reward, timesteps = self.run(org)
            org.update_fitness(total_reward)

            if max_fitness is None or total_reward > max_fitness:
                max_fitness = total_reward
                best_org = org
        if self.td3ga.replay_buffer.size >= self.args.learning_starts:
            self.population.evolve()
         

        
        return max_fitness