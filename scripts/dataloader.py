from torch.utils.data import Dataset
import numpy as np
import torch
import os

class SamaplesDataset(Dataset):
	def __init__(self, dataset_path, file_name):
		self.keys = ["action", "next_state", "not_done", "reward", "state"]
		self.samples = {}
		for key in self.keys:
			self.samples[key] = np.load(os.path.join(dataset_path, file_name + "_" + key + ".npy"))

	def get_samples(self, num_samples):
		idx = np.random.randint(0, len(self.samples['state']), num_samples)
		return {key: self.samples[key][idx] for key in self.keys}

	def __len__(self):
		return len(self.samples['state'])
	
	def __getitem__(self, index):
		return {key: self.samples[key][index] for key in self.keys}
	
	
	
class TrajectoryDataset(Dataset):
	def __init__(self, dataset_path, file_name):
		self.keys = ["action", "next_state", "not_done", "reward", "state"]
		samples = {}
		for key in self.keys:
			samples[key] = np.load(os.path.join(dataset_path, file_name + "_" + key + ".npy"))
		self.Trajectories = []
		Traj = []
		print(len(samples['state']))
		lengths = 0
		Accumulated_Reward = 0
		for i in range(len(samples['state'])):
			action = samples['action'][i]
			state = samples['state'][i]
			next_state = samples['next_state'][i]
			not_done = samples['not_done'][i]
			reward = samples['reward'][i]
			Accumulated_Reward += reward
			Traj.append((action, next_state, not_done, reward, state))
			if not_done == 0:
				self.Trajectories.append(Traj)
				lengths += len(Traj)
				Traj = []
		print("Average Trajectory Length: {}".format(lengths/len(self.Trajectories)))
		print("Average Accumulated Reward: {}".format(Accumulated_Reward/len(self.Trajectories)))
	def __getitem__(self, index):
		return self.Trajectories[index]
	def __len__(self):
		return len(self.Trajectories)
