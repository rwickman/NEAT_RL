import gym
import torch
from neat_rl.rl.sac import SAC

class EnvironmentSAC:
    def __init__(self, args, num_episodes=5000):
        self.args = args
        self.num_episodes=num_episodes

        if self.args.render:
            self.env = gym.make(self.args.env, render_mode='human')
        else:
            self.env = gym.make(self.args.env)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0] 
        self.sac = SAC(self.args, state_dim, action_dim, self.env.action_space)


        if self.args.load:
            self.sac.load()
    

    def run(self, render=False):
        state, _ = self.env.reset()
        done = False
        truncated = False
        total_reward = 0
        cur_step = 0

        while not done and not truncated:
            cur_step += 1
            if self.sac.replay_buffer.size < self.args.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.sac.select_action(state)

            next_state, reward, done, truncated, info = self.env.step(action)

            self.sac.replay_buffer.add(state, action, next_state, reward, done)
            if self.sac.replay_buffer.size >= self.args.learning_starts:
                self.sac.train()
            
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
            print(f"TOTAL REWARD {total_reward} FOR EPISODE {i} TOTAL TIMESTEPS {total_timesteps}")
