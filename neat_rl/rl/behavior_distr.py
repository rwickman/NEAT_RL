import torch
from torch.distributions import Normal
import numpy as np

# 1. Create behavior probabilities
# 2. Sample behaviors
# 3. Train

class BehaviorDistr:
    def __init__(self, args):
        self.args = args
        self.behaviors = None
        self.sample_probs = None 

    def _relative_prob(self, log_probs):
        # print("SUBTRACTED MEAN LOG PROBS:", (log_probs - log_probs.mean())[:20])
        relative_probs = torch.exp(torch.clamp(log_probs - log_probs.mean(), max=40))
        probs = relative_probs/relative_probs.sum()
        return probs

    def _skew(self, probs):
        w = probs ** self.args.skew_val
        z = w.sum()
        skew_weights = w/z
        return w / z

    def sample(self):
        indices = np.random.choice(
            len(self.sample_probs),
            self.args.batch_size,
            p=self.sample_probs)
        
        sample_behaviors = self.behaviors[indices]
        sample_species_id = self.species_ids[indices]

        return sample_behaviors, sample_species_id 


    def refresh(self, behaviors, species_ids):
        self.behaviors = behaviors
        self.species_ids = species_ids
        self.sample_probs, _ = self.behavior_weights(behaviors, behaviors.mean(0), behaviors.std(0) + 1e-6)
        self.sample_probs = self.sample_probs.numpy()
        # print("self.sample_probs.max()", self.sample_probs.max(), self.sample_probs.min(), self.sample_probs.mean())
        # print("self.sample_probs", self.sample_probs[:10])
        # print("self.behaviors", self.behaviors[:5])

    def behavior_weights(self, behaviors, mean, std):
        # print("behavior_mean", mean)
        # print("behavior_std", std)

        # Fit distribution on the behaviors
        distr = Normal(mean, std)
        
        # Get the probabilities of each behavior
        log_probs = distr.log_prob(behaviors).sum(dim=1)
        # print("log_probs[:20]", log_probs[:20])
        skewed_log_probs = log_probs * self.args.skew_val
        skewed_probs = self._relative_prob(skewed_log_probs)
        # probs = self._relative_prob(log_probs)

        assert not torch.any(torch.isnan(skewed_probs))
        assert not torch.any(torch.isinf(skewed_probs))


        return skewed_probs, skewed_probs