import gym
import torch
from neat_rl.rl.td3 import TD3

class Environment:
    def __init__(self, args, num_episodes=5000):
        self.args = args
        self.num_episodes=num_episodes
        self.env = gym.make(self.args.env)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0] 
        max_action = float(self.env.action_space.high[0])
        self.args.policy_noise = self.args.policy_noise * max_action
        self.args.noise_clip = self.args.noise_clip * max_action

        self.td3 = TD3(self.args, state_dim, action_dim, max_action)


        if self.args.load:
            self.td3.load()



    def run(self, render=False):
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
            if self.td3.replay_buffer.size < self.args.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.td3.sample_action(state)

            next_state, reward, done, truncated, info = self.env.step(action)    
            if cur_step > self.args.max_timesteps:
                done = True

            self.td3.replay_buffer.add(state, action, next_state, reward, done)
            if self.td3.replay_buffer.size >= self.args.learning_starts:
                self.td3.train()


            
            if render:
                self.env.render()     
            total_reward += reward
            state = next_state

        return total_reward, cur_step

    def train(self):
        total_timesteps = 0
        for i in range(self.num_episodes):
            total_reward, timesteps = self.run()
            #total_reward, timesteps = self.run(True)
            total_timesteps += timesteps

            #total_reward = self.run(False, True)
            print(f"TOTAL REWARD {total_reward} FOR EPISODE {i}")
            # if i >= 10:
            #     for _ in range(16):
            #         self.td3.train(self.td3.replay_buffer)
                
            if i % 32 == 0: 
                self.td3.save()
                print("SAVED MODEL")
        
        self.td3.save()