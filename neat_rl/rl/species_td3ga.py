import torch
from neat_rl.rl.species_td3 import SpeciesTD3
import numpy as np



class SpeciesTD3GA(SpeciesTD3):
    """TD3 for updating genetic algorithms.""" 
    def __init__(self, args, state_dim, action_dim, max_action, behavior_dim):
        super().__init__(args, state_dim, action_dim, max_action, behavior_dim)
    
    def sample_action_net(self, net, state, evaluate=False):
        action_org = self.select_action_net(net, state)
        if evaluate:
            return action_org, action_org
        else: 
            action = (
                action_org
                    + np.random.normal(0, self.max_action * self.args.expl_noise, size=self.action_dim)
                ).clip(-self.max_action, self.max_action)
            return action, action_org
        
    def select_action_net(self, net, state): 
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = net(state).cpu().data.numpy().flatten()
        return action

    def pg_update(self, net, species_id):
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.org_lr)
        species_ids = torch.full((self.args.batch_size,), species_id, device=self.device)
        for _ in range(self.args.n_org_updates):
            state  = self.replay_buffer.sample_states(self.args.batch_size, species_id)
            
            actor_loss = -self.critic.Q1(state, net(state), species_ids).mean()
            optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.args.max_norm)
            optimizer.step()