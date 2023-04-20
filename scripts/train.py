import torch
import numpy as np
from scripts.dataloader import D4RLTrajectoryDataset
from scripts.args import get_args
from agents.bc.bc_agent import BC
from torch.utils.data import DataLoader
def train():
	pass

if __name__ == "__main__":
	args = get_args()
	dataset = D4RLTrajectoryDataset(args.dataset_path, args.file_name)
	data_loader = DataLoader(
		dataset,
		bacth_size=args.batch_size,
		shuffle = True
	)
	model = BC(state_dim=11, action_dim=3, hidden_dim=128)
	train(args, model, data_loader)