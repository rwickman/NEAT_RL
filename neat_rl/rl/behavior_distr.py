import torch
from torch.distributions import Normal
import numpy as np
import json, os
# 1. Create behavior probabilities
# 2. Sample behaviors
# 3. Train

class BehaviorDistr:
    def __init__(self, args, behavior_dim, max_size=2048, batch_size=64):
        self.batch_size = batch_size
        self.args = args
        self.max_size = max_size
        self.behaviors = np.zeros((self.max_size, behavior_dim))
        self.species_ids = np.zeros((self.max_size, 1))
        self.ptr = 0
        self.size = 0
        self.save_file = os.path.join(self.args.save_dir, "behaviors.json")

        self.sample_probs = None 

    def _relative_prob(self, log_probs):
        relative_probs = torch.exp(torch.clamp(log_probs - log_probs.mean(), max=30))
        probs = relative_probs/relative_probs.sum()
        return probs

    def _skew(self, probs):
        w = probs ** self.args.skew_val
        z = w.sum()
        skew_weights = w/z
        return w / z

    def add(self, behavior, species_id):
        self.behaviors[self.ptr] = behavior
        self.species_ids[self.ptr] = species_id
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self):
        indices = np.random.choice(
            len(self.sample_probs),
            self.batch_size,
            p=self.sample_probs,
            replace=False)
        
        sample_behaviors = self.behaviors[indices]
        sample_species_id = self.species_ids[indices]

        return torch.FloatTensor(sample_behaviors), torch.LongTensor(sample_species_id).view(-1) 

    def save(self):
        save_dict = {
            "behaviors": self.behaviors.tolist(),
            "species_ids": self.species_ids.tolist(),
            "size": self.size,
            "ptr": self.ptr
        }

        with open(self.save_file, "w") as f:
            json.dump(save_dict, f)
    
    def load(self):
        if os.path.exists(self.save_file):

            with open(self.save_file) as f:
                save_dict = json.load(f)
            
            self.behaviors[:len(save_dict["behaviors"])] = np.array(save_dict["behaviors"])[:self.max_size]
            self.species_ids[:len(save_dict["species_ids"])] = np.array(save_dict["species_ids"])[:self.max_size]
            self.size = max(save_dict["size"], self.size)
            self.ptr = save_dict["ptr"] % self.max_size




    def refresh(self):
        behaviors = torch.tensor(self.behaviors[:self.size])
        self.sample_probs, _ = self.behavior_weights(behaviors, behaviors.mean(0), behaviors.std(0))
        self.sample_probs = self.sample_probs.numpy()
        print("self.sample_probs", self.sample_probs[:10])
        print("self.behaviors", self.behaviors[:10])

    def behavior_weights(self, behaviors, mean, std):
        print("behavior_mean", mean)
        print("behavior_std", std)

        # Fit distribution on the behaviors
        distr = Normal(mean, std)
        
        # Get the probabilities of each behavior
        log_probs = distr.log_prob(behaviors).sum(dim=1)
        skewed_log_probs = log_probs * self.args.skew_val
        skewed_probs = self._relative_prob(skewed_log_probs)
        probs = self._relative_prob(log_probs)

        assert not torch.any(torch.isnan(skewed_probs))
        assert not torch.any(torch.isinf(skewed_probs))


        return skewed_probs, probs