from torch.utils.data import Dataset
import numpy as np
import torch
import os

class D4RLTrajectoryDataset(Dataset):
	def __init__(self, dataset_path, file_name):
		if dataset_path == "":
			return
		self.keys = ["action", "next_state", "not_done", "reward", "state"]
		self.samples = {}
		for key in self.keys:
			self.samples[key] = np.load(os.path.join(dataset_path, file_name + "_" + key + ".npy"))

	@staticmethod
	def from_buffer(buffer):
		dataset = D4RLTrajectoryDataset("", "")
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

		return dataset

	def __len__(self):
		return len(self.samples['state'])
	
	def __getitem__(self, index):
		return {key: self.samples[key][index] for key in self.keys}
