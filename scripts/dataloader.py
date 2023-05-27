from torch.utils.data import Dataset
import numpy as np
import torch
import os

class SamaplesDataset(Dataset):
	def __init__(self, dataset_path, file_name):
		if dataset_path == "":
			return
		self.keys = ["state", "next_state", "action",  "reward" , "not_done"]
		self.samples = {}
		for key in self.keys:
			self.samples[key] = np.load(os.path.join(dataset_path, file_name + "_" + key + ".npy"))

	def get_samples(self, num_samples):
		idx = np.random.randint(0, len(self.samples['state']), num_samples)
		return {key: self.samples[key][idx] for key in self.keys}
	@staticmethod
	def from_buffer(buffer):
		dataset = SamaplesDataset("", "")
		dataset.keys = buffer.keys
		dataset.samples = {}
		for sample in buffer.storage:
			for i, key in enumerate(dataset.keys):
				if sample[i] is None:
					continue
				if key not in dataset.samples:
					dataset.samples[key] = []
				dataset.samples[key].append(sample[i])
		dataset.keys = list(dataset.samples.keys())
		for key in dataset.keys:
			dataset.samples[key] = np.array(dataset.samples[key])
		if "done" in dataset.samples:
			dataset.samples["not_done"] = 1 - dataset.samples["done"]
			dataset.keys.append("not_done")
		return dataset
	@staticmethod
	def from_traj(Trajectory):
		dataset = SamaplesDataset("", "")
		dataset.keys = Trajectory.keys
		dataset.samples = {}
		for Traj in Trajectory.Trajectories:
			for pair in Traj:
				for i, key in enumerate(dataset.keys):
					if key not in dataset.samples:
						dataset.samples[key] = []
					dataset.samples[key].append(pair[i])
				
		return dataset

	def __len__(self):
		return len(self.samples['state'])
	
	def __getitem__(self, index):
		return {key: self.samples[key][index] for key in self.keys}

class TrajectoryDataset(Dataset):
	def __init__(self, dataset_path, file_name, truncation = 0, Threshold = 0):
		self.keys = ["state", "next_state", "action",  "reward" , "not_done"]
		samples = {}
		for key in self.keys:
			samples[key] = np.load(os.path.join(dataset_path, file_name + "_" + key + ".npy"))
		self.Trajectories = []
		Traj = []
		print(len(samples['state']))
		lengths = 0
		Mn_len = 10000
		Mx_len = 0 
		Accumulated_Reward = 0
		for i in range(len(samples['state'])):
			action = samples['action'][i]
			state = samples['state'][i]
			next_state = samples['next_state'][i]
			not_done = samples['not_done'][i]
			reward = samples['reward'][i]
			Accumulated_Reward += reward
			Traj.append((state, next_state, action, reward, not_done))
			if not_done == 0:
				Mx_len = max(Mx_len, len(Traj))
				Mn_len = min(Mn_len, len(Traj))
				if (len(Traj)<=Threshold):
					Traj = []
					continue
				lengths += len(Traj)
				
				if truncation > 0:
					Traj = Traj[:-int(truncation * len(Traj))]
				self.Trajectories.append(Traj)
				Traj = []
		print("Before Truncation:")
		print("Max Trajectory Length: {}".format(Mx_len))
		print("Min Trajectory Length: {}".format(Mn_len))
		print("Average Trajectory Length: {}".format(lengths/len(self.Trajectories)))
		print("Average Accumulated Reward: {}".format(Accumulated_Reward/len(self.Trajectories)))


	def get_samples(self, num_samples):
		idx = np.random.randint(0, len(self.Trajectories), num_samples)
		return self.Trajectories[idx]
	
	def __getitem__(self, index):
		return self.Trajectories[index]
	def __len__(self):
		return len(self.Trajectories)
