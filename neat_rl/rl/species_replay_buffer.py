
import numpy as np
import torch
import os

class SpeciesReplayBuffer:
	def __init__(self, state_dim, action_dim, behavior_dim, max_size=int(1e6), resample_species=False):
		self.max_size = max_size
		self.resample_species = resample_species
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.action_org = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.species_id = np.zeros((max_size, 1))
		self.species_id[:] = -1
		self.behavior = np.zeros((max_size, behavior_dim))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, action_org, next_state, reward, species_id, behavior, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.action_org[self.ptr] = action_org
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.species_id[self.ptr] = species_id
		self.behavior[self.ptr] = behavior
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.choice(self.size, size=batch_size, replace=False)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.action_org[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.LongTensor(self.species_id[ind]).to(self.device).view(-1),
			torch.FloatTensor(self.behavior[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

	def sample_states(self, batch_size, species_id):
		if self.resample_species:
			ind = np.random.choice(self.size, size=batch_size, replace=False)
		else:
			species_states = np.where(self.species_id[:self.size] == species_id)[0]
			ind = np.random.choice(species_states, size=batch_size)

		return torch.FloatTensor(self.state[ind]).to(self.device)

	def save(self, save_dir):
		out_file = os.path.join(save_dir, "replay_buffer.npz")
		d = {
			"state": self.state,
			"action": self.action,
			"next_state": self.next_state,
			"reward": self.reward,
			"species_id": self.species_id,
			"not_done": self.not_done,
			"ptr": self.ptr,
			"size": self.size
		}

		np.savez(out_file, **d)

	def load(self, save_dir):
		out_file = os.path.join(save_dir, "replay_buffer.npz")
		data = np.load(out_file)

		self.state = data["state"]
		self.action = data["action"]
		self.next_state = data["next_state"]
		self.species_id = data["species_id"]
		self.not_done = data["not_done"]
		self.ptr = int(data["ptr"])
		self.size = int(data["size"])

