from torch.utils.data import Dataset
import numpy as np
import torch
import os

class D4RLTrajectoryDataset(Dataset):
	def __init__(self, dataset_path, file_name):
		self.keys = ["action", "next_state", "not_done", "reward", "state"]
		self.samples = {}
		for key in self.keys:
			self.samples[key] = np.load(os.path.join(dataset_path, file_name + "_" + key + ".npy"))
        

	def __len__(self):
		return len(self.samples['state'])
	
	def __getitem__(self, index):
		return {key: self.samples[key][index] for key in self.keys}