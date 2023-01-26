import torch
from torch.distributions import Normal
import numpy as np

# 1. Create behavior probabilities
# 2. Sample behaviors
# 3. Train

class BehaviorDistr:
    def __init__(self, args, behavior_dim):
        self.args = args
        self.max_size = self.args.behavior_capacity
        self.behaviors = np.zeros((self.max_size, behavior_dim))
        self.species_ids = np.zeros((self.max_size, ))
        self.size = 0
        self.ptr = 0
        self.sample_probs = None 
        self.skewed_distr = None

    def _relative_prob(self, log_probs):
        # print("SUBTRACTED MEAN LOG PROBS:", (log_probs - log_probs.mean())[:20])
        relative_probs = torch.exp(torch.clamp(log_probs - log_probs.mean(), max=40))
        probs = relative_probs/relative_probs.sum()
        return probs

    def add(self, behavior, species_id):
        self.species_ids[self.ptr] = species_id
        self.behaviors[self.ptr] = behavior
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _skew(self, probs):
        w = probs ** self.args.skew_val
        z = w.sum()
        skew_weights = w/z
        return w / z

    def get_log_prob(self, behavior):
        if self.skewed_distr is not None:
            return self.skewed_distr.log_prob(behavior).sum(dim=1).item()
        else:
            return 0

    def sample(self):
        indices = np.random.choice(
            len(self.sample_probs),
            self.args.batch_size,
            p=self.sample_probs)
        
        sample_behaviors = torch.FloatTensor(self.behaviors[indices].tolist())
        sample_species_id = torch.LongTensor(self.species_ids[indices].tolist())

        # # Resample behaviors
        if self.args.resample_behavior:
            resample_prob = torch.zeros(self.args.batch_size).uniform_()
            sampled_skewed_behaviors = self.skewed_distr.sample(torch.Size([self.args.batch_size]))
            sample_behaviors[resample_prob <= self.args.resample_behavior_prob] = sampled_skewed_behaviors[resample_prob <= self.args.resample_behavior_prob] 

        return sample_behaviors, sample_species_id 

    def refresh(self):
        behaviors = torch.tensor(self.behaviors[:self.size], dtype=torch.float32)
        self.sample_probs, _ = self.behavior_weights(behaviors, behaviors.mean(0), behaviors.std(0) + 1e-6)
        self.sample_probs = self.sample_probs.numpy()
    
    def behavior_weights(self, behaviors, mean, std):
        # Fit distribution on the behaviors
        distr = Normal(mean, std)
        
        # Get the probabilities of each behavior
        log_probs = distr.log_prob(behaviors).sum(dim=1)
        # print("log_probs[:20]", log_probs[:20])
        skewed_log_probs = log_probs * self.args.skew_val
        skewed_probs = self._relative_prob(skewed_log_probs)
        
        skewed_mean = skewed_probs.matmul(behaviors)
        # print("skewed_mean", skewed_mean, "old_mean", mean)
        skewed_std = ((behaviors - skewed_mean) ** 2).mean(dim=0) ** 0.5
        # print("skewed_std", skewed_std, "old std", std)
        self.skewed_distr = Normal(skewed_mean, skewed_std + 1e-6)

        
        
        # probs = self._relative_prob(log_probs)

        assert not torch.any(torch.isnan(skewed_probs))
        assert not torch.any(torch.isinf(skewed_probs))


        return skewed_probs, skewed_probs