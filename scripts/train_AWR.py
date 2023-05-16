import torch
import numpy as np

import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))

import copy
from evaluate import evaluation
from dataloader import SamaplesDataset
from args import get_args
from agents.AWR.AWR_agent import AWR
from torch.utils.data import DataLoader
from tqdm import *
from torch.utils.tensorboard import SummaryWriter
def compute_advantage(gamma, lmbda, td_delta):
	td_delta = td_delta.detach().numpy()
	advantage_list = []
	advantage = 0.0
	for delta in td_delta[::-1]:
		advantage = gamma * lmbda * advantage + delta
		advantage_list.append(advantage)
	advantage_list.reverse()
	return torch.tensor(advantage_list, dtype=torch.float)

def train(model, dataset, args):
	actor_optim = torch.optim.Adam(model.actor.parameters(), lr=5e-5)
	critic_optim = torch.optim.Adam(model.critic.parameters(), lr=1e-4)
	writer = SummaryWriter()
	step = 0
	Samples_per_iter = 256
	for epoch in trange(args.n_epochs):
		batch_data = dataset.get_samples(Samples_per_iter)
		states = torch.tensor(batch_data['state'], dtype=torch.float).to(args.device)
		next_states = torch.tensor(batch_data['next_state'], dtype=torch.float).to(args.device)
		actions = torch.tensor(batch_data['action'], dtype=torch.float).to(args.device)
		rewards = torch.tensor(batch_data['reward'], dtype=torch.float).to(args.device)
		not_dones = torch.tensor(batch_data['not_done'], dtype=torch.float).to(args.device)
		perm = torch.randperm(states.shape[0])
		tmp_step = step
		with tqdm(total = states.shape[0]//args.batch_size) as pbar:
			for i in range(0, states.shape[0], args.batch_size):
				batch_states = states[perm[i:i+args.batch_size]]
				batch_next_states = next_states[perm[i:i+args.batch_size]]
				batch_actions = actions[perm[i:i+args.batch_size]]	
				batch_rewards = rewards[perm[i:i+args.batch_size]]	
				batch_not_dones = not_dones[perm[i:i+args.batch_size]]
				td_target = batch_rewards + model.gamma * model.critic(batch_next_states).detach() * batch_not_dones
				critic_Loss = torch.nn.functional.mse_loss(model.critic(batch_states), td_target, reduction='mean')
				critic_optim.zero_grad()
				critic_Loss.backward()
				critic_optim.step()
				pbar.set_description("Epoch: {},Training Critic".format(epoch))
				pbar.set_postfix(critic_loss=critic_Loss.item())
				pbar.update(1)
				writer.add_scalar("critic_loss", critic_Loss.item(), step)
				step += 1

		step = tmp_step
		with tqdm(total = states.shape[0]//args.batch_size) as pbar:
			for i in range(0, states.shape[0], args.batch_size):
				batch_states = states[perm[i:i+args.batch_size]]
				batch_next_states = next_states[perm[i:i+args.batch_size]]
				batch_actions = actions[perm[i:i+args.batch_size]]	
				batch_rewards = rewards[perm[i:i+args.batch_size]]	
				batch_not_dones = not_dones[perm[i:i+args.batch_size]]
				td_target = batch_rewards + model.gamma * model.critic(batch_next_states).detach() * batch_not_dones
				td_delta = td_target - model.critic(batch_states).detach()
				advantage = compute_advantage(model.gamma, model.lamb, td_delta)
				weights = torch.clamp(torch.exp(advantage/model.beta), max = 20)
				actor_loss =  torch.mean( torch.sum((model.actor(batch_states) - batch_actions)**2, dim=1) * weights)
				actor_optim.zero_grad()
				actor_loss.backward()
				actor_optim.step()

				pbar.set_description("Epoch: {},Training Actor".format(epoch))
				pbar.set_postfix(actor_loss=actor_loss.item())
				pbar.update(1)
				writer.add_scalar("actor_loss", actor_loss.item(), step)
				step += 1

		# with tqdm(total = len(dataLoader)) as pbar:
		# 	for batch in dataLoader:
		# 		# Get data
		# 		state = batch['state'].float().to(args.device)
		# 		action = batch['action'].float().to(args.device)
		# 		Reward = batch['reward'].float().to(args.device)
		# 		next_state = batch['next_state'].float().to(args.device)
		# 		not_done = batch['not_done'].float().to(args.device)

		# 		# advantage = Reward - model.critic(state).detach()
				
		# 		td_target = Reward + model.gamma * old_model.critic(next_state).detach() * not_done
		# 		td_delta = td_target - old_model.critic(state).detach()
		# 		advantage = compute_advantage(model.gamma, model.lamb, td_delta)
		# 		critic_Loss = torch.nn.functional.mse_loss(model.critic(state), td_target, reduction='mean')


		# 		critic_optim.zero_grad()
		# 		critic_Loss.backward()
		# 		critic_optim.step()
		# 		# import pdb
		# 		# pdb.set_trace()
		# 		weight = torch.clamp(torch.exp(1/model.beta * advantage), max=20)
		# 		actor_loss = ((model.actor(state) - action)**2) * weight

		# 		actor_loss = torch.mean(actor_loss)
		# 		actor_optim.zero_grad()
		# 		actor_loss.backward()
		# 		actor_optim.step()
		# 		pbar.set_description("Epoch: {}".format(epoch))
		# 		pbar.set_postfix(actor_loss=actor_loss.item(), critic_loss=critic_Loss.item(), weight = weight[0], advantage = advantage[0])
		# 		pbar.update(1)
		# 		writer.add_scalar("actor_loss", actor_loss.item(), step)
		# 		writer.add_scalar("critic_loss", critic_Loss.item(), step)
		# 		writer.add_scalar("weight", weight[0], step)
		# 		writer.add_scalar("advantage", advantage[0], step)
		# 		step += 1
				

		Reward, episodes_len = evaluation(model)
		torch.save(model, os.path.join(args.save_dir,"AWR_{}.pth".format(epoch%10)))
		# tqdm.set_description("Epoch: {}, Reward: {}".format(epoch, Reward))
		print("###############################################")
		print("Epoch: {}, Reward: {}, Mean Episodes Length: {}".format(epoch, Reward, episodes_len))
	pass

if __name__ == "__main__":
	args = get_args()
	dataset = SamaplesDataset(args.dataset_path, args.file_name)
	# dataLoader = DataLoader(
	# 	dataset,
	# 	batch_size=args.batch_size,
	# 	shuffle = True
	# )
	model = AWR(state_dim=11, action_dim=3).to(args.device)
	train(model, dataset, args)