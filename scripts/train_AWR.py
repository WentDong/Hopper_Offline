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
from agents.AWR.AWR_agent import AWR
from torch.utils.data import DataLoader
from tqdm import *

def compute_advantage(gamma, lmbda, td_delta):
	td_delta = td_delta.detach().numpy()
	advantage_list = []
	advantage = 0.0
	for delta in td_delta[::-1]:
		advantage = gamma * lmbda * advantage + delta
		advantage_list.append(advantage)
	advantage_list.reverse()
	return torch.tensor(advantage_list, dtype=torch.float)

def train(model, dataLoader, args):
	actor_optim = torch.optim.Adam(model.actor.parameters(), lr=5e-5)
	critic_optim = torch.optim.Adam(model.critic.parameters(), lr=1e-4)
	
	for epoch in trange(args.n_epochs):
		with tqdm(total = len(dataLoader)) as pbar:
			for batch in dataLoader:
				# Get data
				state = batch['state'].float().to(args.device)
				action = batch['action'].float().to(args.device)
				Reward = batch['reward'].float().to(args.device)
				next_state = batch['next_state'].float().to(args.device)
				not_done = batch['not_done'].float().to(args.device)

				advantage = Reward - model.critic(state).detach()
				
				td_target = Reward + model.gamma * model.critic(next_state).detach() * not_done
				td_delta = td_target - model.critic(state).detach()
				advantage = compute_advantage(model.gamma, model.lamb, td_delta)
				critic_Loss = torch.nn.functional.mse_loss(model.critic(state), td_target, reduction='mean')

				critic_optim.zero_grad()
				critic_Loss.backward()
				critic_optim.step()
				# import pdb
				# pdb.set_trace()
				weight = torch.clamp(torch.exp(1/model.beta * advantage), max=20)
				
				actor_loss = ((model.actor(state) - action)**2) * weight

				actor_loss = torch.mean(actor_loss)
				actor_optim.zero_grad()
				actor_loss.backward()
				actor_optim.step()
				pbar.set_description("Epoch: {}".format(epoch))
				pbar.set_postfix(actor_loss=actor_loss.item(), critic_loss=critic_Loss.item(), weight = weight[0])
				pbar.update(1)
				

		Reward, episodes_len = evaluation(model)
		torch.save(model, os.path.join(args.save_dir,"AWR_{}.pth".format(epoch%10)))
		# tqdm.set_description("Epoch: {}, Reward: {}".format(epoch, Reward))
		print("Epoch: {}, Reward: {}, Mean Episodes Length: {}".format(epoch, Reward, episodes_len))
	pass

if __name__ == "__main__":
	args = get_args()
	dataset = D4RLTrajectoryDataset(args.dataset_path, args.file_name)
	dataLoader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle = True
	)
	model = AWR(state_dim=11, action_dim=3).to(args.device)
	train(model, dataLoader, args)