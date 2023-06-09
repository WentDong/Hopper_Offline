import torch
import numpy as np

import json
import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))

import copy
from evaluate import *
from dataloader import TrajectoryDataset
from args import get_args
from agents.AWR.AWR_agent import AWR
from scripts.utils import Mount_Carlo_Estimation, Traj_Replay_Buffer, plot_eval, compute_advantage
# from agents.bail import utils
from torch.utils.data import DataLoader
from tqdm import *
from torch.utils.tensorboard import SummaryWriter

def train(model, replay_buffer, args):
	actor_optim = torch.optim.AdamW(model.actor.parameters(), lr=args.lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(actor_optim, 'min', patience=2000, factor=0.7, verbose=True)
	critic_optim = torch.optim.AdamW(model.critic.parameters(), lr=args.lr)
	writer = SummaryWriter()
	step = 0	
	Trajs_per_iter = 128
	Eval = Evaluator(device=args.device)
	idx = 0
	Mx_Reward = 0
	if args.plot:
		step_interval = args.plot_interval
		Reward_log = []
	dir = os.path.join(args.save_dir, "AWR", str(idx))
	while os.path.exists(dir):
		idx += 1
		dir = os.path.join(args.save_dir, "AWR", str(idx))
	os.makedirs(dir)
	with open(os.path.join(dir, "args.json"), "w") as f:
		json.dump(args.__dict__, f, indent=2)
	for epoch in trange(args.n_epochs):
		start = 0
		Returns, Trajs = Mount_Carlo_Estimation(replay_buffer, Sample_Size = Trajs_per_iter, discount_factor = model.gamma) 
		with tqdm(total=Trajs_per_iter) as pbar:
			for _ in range(Trajs_per_iter//args.batch_size):
				batch_traj = Trajs[start:start+args.batch_size]
				batch_return = Returns[start:start+args.batch_size]
				start += args.batch_size

				for i in range(len(batch_traj)):
					traj = batch_traj[i]
					returns = batch_return[i]
					returns = torch.tensor(returns, device = args.device).float().unsqueeze(1)
					# print(traj[0][0])
					states = [traj[i][0] for i in range(len(traj))]
					states = torch.tensor(states, device = args.device).float()
					# print(states.shape)
					next_states = [traj[i][1] for i in range(len(traj))]
					next_states = torch.tensor(next_states, device = args.device).float()
					# print(next_states.shape)
					actions = [traj[i][2] for i in range(len(traj))]
					actions = torch.tensor(actions, device = args.device).float()
					rewards = [traj[i][3] for i in range(len(traj))]
					rewards = torch.tensor(rewards, device = args.device).float()
					dones = [traj[i][4] for i in range(len(traj))]
					dones = torch.tensor(dones, device = args.device).float()
					# print(rewards.shape, dones.shape, actions.shape)

					td_target = rewards + model.gamma * model.critic(next_states).detach() * (1-dones)
					td_delta = td_target - model.critic(states).detach()
					advantage = compute_advantage(model.gamma, model.lamb, td_delta)

					critic_loss = torch.nn.functional.mse_loss(model.critic(states), td_target, reduction='mean')
					# critic_loss = torch.nn.functional.mse_loss(model.critic(states), returns, reduction='mean')
					critic_optim.zero_grad()
					critic_loss.backward()
					critic_optim.step()
					
					
					# td_target = returns
					# advantage = td_target - model.critic(states).detach() 
					weight = torch.clamp(torch.exp(1/model.beta * advantage), max=20)
					actor_loss = ((model.actor(states) - actions)**2) * weight
					actor_loss = torch.mean(actor_loss, dim = 1)
					actor_loss = torch.mean(actor_loss)
					actor_optim.zero_grad()
					actor_loss.backward()
					actor_optim.step()
					scheduler.step(actor_loss)
					writer.add_scalar('actor_loss', actor_loss, step)
					writer.add_scalar('critic_loss', critic_loss, step)
					pbar.set_description("Epoch: {}".format(epoch))
					pbar.set_postfix(actor_loss=actor_loss.item(), critic_loss=critic_loss.item(), weight = weight[0].item(), advantage = advantage[0].item())
					pbar.update(1)
					if args.plot and ((step+len(traj)) // step_interval)-step//step_interval>0:
						# import pdb; pdb.set_trace()
						# print(step)
						Reward, _ = Eval.evaluate(model)
						Reward_log.append(Reward)
					step += len(traj)


			# for batch in dataLoader:
			# 	# Get data
			# 	print(batch.keys())
			# 	print(batch['state'].shape)	
			# 	state = batch['state'].float().to(args.device)
			# 	action = batch['action'].float().to(args.device)
			# 	Reward = batch['reward'].float().to(args.device)
			# 	next_state = batch['next_state'].float().to(args.device)
			# 	done = batch['done'].float().to(args.device)

			# 	# advantage = Reward - model.critic(state).detach()
	
			# 	td_target = Reward + model.gamma * model.critic(next_state).detach() * (1-done)
			# 	td_delta = td_target - model.critic(state).detach()
			# 	advantage = compute_advantage(model.gamma, model.lamb, td_delta)
			# 	critic_Loss = torch.nn.functional.mse_loss(model.critic(state), td_target, reduction='mean')


			# 	critic_optim.zero_grad()
			# 	critic_Loss.backward()
			# 	critic_optim.step()
			# 	# import pdb
			# 	# pdb.set_trace()
			# 	weight = torch.clamp(torch.exp(1/model.beta * advantage), max=20)
			# 	actor_loss = ((model.actor(state) - action)**2) * weight

			# 	actor_loss = torch.mean(actor_loss)
			# 	actor_optim.zero_grad()
			# 	actor_loss.backward()
			# 	actor_optim.step()
			# 	pbar.set_description("Epoch: {}".format(epoch))
			# 	pbar.set_postfix(actor_loss=actor_loss.item(), critic_loss=critic_Loss.item(), weight = weight[0], advantage = advantage[0])
			# 	pbar.update(1)
			# 	writer.add_scalar("actor_loss", actor_loss.item(), step)
			# 	writer.add_scalar("critic_loss", critic_Loss.item(), step)
			# 	writer.add_scalar("weight", weight[0], step)
			# 	writer.add_scalar("advantage", advantage[0], step)
			# 	step += 1
				
		Reward, episodes_len = Eval.evaluate(model)
		torch.save(model.state_dict(), os.path.join(dir, "AWR_{}.pth".format(epoch%10)))
		if Reward > Mx_Reward:
			Mx_Reward = Reward
			torch.save(model.state_dict(), os.path.join(dir, "AWR_best.pth"))
		# tqdm.set_description("Epoch: {}, Reward: {}".format(epoch, Reward))
		print("###############################################")
		with open(os.path.join(dir, "log.txt"), "a") as f:
			f.write("Epoch: {}, Reward: {}, Mean Episodes Length: {}\n".format(epoch, Reward, episodes_len))
		print("Epoch: {}, Reward: {}, Mean Episodes Length: {}".format(epoch, Reward, episodes_len))
	if args.plot:
		return Reward_log

if __name__ == "__main__":
	args = get_args("bail")
	Traj_dataset = TrajectoryDataset(args.dataset_path, args.file_name, args.trajectory_truncation, Threshold=args.len_threshold)
	replay_buffer = Traj_Replay_Buffer()
	replay_buffer.load(Traj_dataset)

	# dataset = SamaplesDataset(args.dataset_path, args.file_name)
	# dataLoader = DataLoader(
	# 	dataset,
	# 	batch_size=args.batch_size,
	# 	shuffle = True
	# )	
	if not args.plot:
		model = AWR(state_dim=11, action_dim=3, hidden_dim=args.hidden_dim).to(args.device)
		train(model, replay_buffer, args)
	else:
		step_interval = args.plot_interval
		Reward_logs = []
		for _ in range(args.training_iteration):
			model = AWR(state_dim=11, action_dim=3, hidden_dim=args.hidden_dim).to(args.device)
			Reward_log = train(model, replay_buffer, args)
			Reward_logs.append(Reward_log)
		
		Reward_logs = np.array(Reward_logs)
		np.save(os.path.join(args.save_dir, "AWR_Rewards.npy"), Reward_logs)
		plot_eval(step_interval, Reward_logs, "AWR")