import torch
from neat_rl.rl.td3 import TD3
import numpy as np



class TD3GA(TD3):
    """TD3 for updating genetic algorithms.""" 
    def __init__(self, args, state_dim, action_dim, max_action):
        super().__init__(args, state_dim, action_dim, max_action)
    

    def sample_action_net(self, net, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = net(state).cpu().data.numpy().flatten() 
        action = (
            action
                + np.random.normal(0, self.max_action * self.args.expl_noise, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
        return action
    

    def pg_update(self, net):
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.actor_lr)
        for _ in range(self.args.n_org_updates):
            state = self.replay_buffer.sample_states(self.args.batch_size)

            actor_loss = -self.critic.Q1(state, net(state)).mean()
            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()
        
        self.critic_optimizer.zero_grad()
