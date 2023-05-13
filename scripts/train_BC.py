import torch
import numpy as np

import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))


from evaluate import evaluation
from dataloader import D4RLTrajectoryDataset
from args import get_args
from agents.bc.bc_agent import BC
from torch.utils.data import DataLoader
from tqdm import *


def train(model, dataLoader, args):
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
	Mx_Reward = 0
	dir = os.path.join(args.save_dir, "BC")
	if not os.path.exists(dir):
		os.makedirs(dir)
	for epoch in trange(args.n_epochs):
		with tqdm(total = len(dataLoader)) as pbar:
			for batch in dataLoader:
				# Get data
				state = batch['state'].float().to(args.device)
				# print(state.shape, type(state))
				action = batch['action'].float().to(args.device)
				loss = model.train(state, action)
				
				optimizer.zero_grad()
				loss.backward()
				# Update parameters
				optimizer.step()
				pbar.set_description("Epoch: {}".format(epoch))
				pbar.set_postfix(loss=loss.item())
				pbar.update(1)
				# Print loss
		Reward, episodes_len = evaluation(model)
		if Reward> Mx_Reward:
			torch.save(model, os.path.join(dir, "BC_best.pth"))
		scheduler.step()
		torch.save(model, os.path.join(dir,"BC_{}.pth".format(epoch%10)))
		# tqdm.set_description("Epoch: {}, Reward: {}".format(epoch, Reward))
		print("Epoch: {}, Reward: {}, Mean Episodes Length: {}".format(epoch, Reward, episodes_len))

if __name__ == "__main__":
	args = get_args()
	dataset = D4RLTrajectoryDataset(args.dataset_path, args.file_name)
	dataLoader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle = True
	)
	model = BC(state_dim=11, action_dim=3, hidden_dim=128).to(args.device)
	train(model, dataLoader, args)