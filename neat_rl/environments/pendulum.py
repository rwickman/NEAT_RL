import gym
import torch
from neat_rl.rl.td3 import TD3
from neat_rl.rl.replay_buffer import ReplayBuffer

class Pendulum:
    def __init__(self, args, num_episodes=5000):
        self.args = args
        self.num_episodes=num_episodes
        self.env = gym.make('Pendulum-v1', g=9.81)
        self.td3 = TD3(self.args, 3, 1, 2.0)
        self.goal = -0.1
        self.replay_buffer = ReplayBuffer(3, 1, 65536)
        self.max_steps = 600
        if self.args.load:
            self.td3.load()



    def run(self, random=False, render=False):
        if render:
            self.env = gym.make('Pendulum-v1', g=9.81, render_mode='human')
        else:
            self.env = gym.make('Pendulum-v1', g=9.81)


        state, _ = self.env.reset()
        done = False
        truncated = False
        total_reward = 0
        cur_step = 0
        while not done and not truncated:
            cur_step += 1
            if random:
                action = self.env.action_space.sample()
            else:
                action = self.td3.sample_action(state)

            next_state, reward, done, truncated, info = self.env.step(action)    
            if cur_step > self.max_steps:
                done = True

            self.replay_buffer.add(state, action, next_state, reward, float(done or truncated))
            
            if render:
                self.env.render()     
            total_reward += reward
            state = next_state

        return total_reward, cur_step

    def train(self):
        total_timesteps = 0
        for i in range(self.num_episodes):
            total_reward, timesteps = self.run(total_timesteps >= self.args.learning_starts)
            #total_reward, timesteps = self.run(False, True)
            total_timesteps += timesteps

            #total_reward = self.run(False, True)
            print(f"TOTAL REWARD {total_reward} FOR EPISODE {i}")
            if i >= 10:
                for _ in range(16):
                    self.td3.train(self.replay_buffer)
                
            if i % 32 == 0: 
                self.td3.save()
                print("SAVED MODEL")
        
        self.td3.save()