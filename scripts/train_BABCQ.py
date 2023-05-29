import numpy as np
import torch

import json
import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))

from evaluate import *
from dataloader import *
from args import get_args
from agents.BAIL import utils
from agents.BAIL.mcret import *
from agents.BAIL.bail_training import Value, train_upper_envelope
from agents.BCQ.bcq_agent import BCQ
from scripts.utils import plot_eval
from torch.utils.data import DataLoader
from tqdm import *
from torch.utils.tensorboard import SummaryWriter
from train_BAIL import get_mcret, train_ue
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_batch_ue(replay_buffer, states, returns, upper_envelope, C, args):
    states = torch.from_numpy(states).to(device)
    returns = torch.from_numpy(returns).to(device)
    upper_envelope = upper_envelope.to(device)

    ratios = []
    for i in range(states.shape[0]):
        s, ret = states[i], returns[i]
        s_val = upper_envelope(s.unsqueeze(dim=0).float()).detach().squeeze()
        ratios.append(ret / torch.min(s_val, C) if C is not None else ret / s_val)

    ratios = torch.stack(ratios).view(-1)
    increasing_ratios, increasing_ratio_indices = torch.sort(ratios)
    bor_ind = increasing_ratio_indices[-int(args.select_percentage * states.shape[0])]
    border = ratios[bor_ind]

    '''begin selection'''
    selected_buffer = utils.ReplayBuffer()
    selected_buffer.keys = replay_buffer.keys + ['select']
    print('Selecting with ue border', border.item())
    selected_len = 0
    for i in range(states.shape[0]):
        rat = ratios[i]
        obs, next_obs, act, reward, done = replay_buffer.index(i)
        selected_buffer.add((obs, next_obs, act, reward, done, (rat >= border).item()))
        selected_len += (rat >= border).item()

    initial_len = replay_buffer.get_length()
    print('border:', border, 'selecting ratio:', selected_len, '/', initial_len)

    return (selected_buffer, selected_len, border)

def train(model, dataLoader, args, algo = "BCQ"):
	eval = Evaluator(device =  args.device)
	Mx_Reward = 0
	idx =  0
	steps = 0
	dir = os.path.join(args.save_dir, algo, str(idx))
	while os.path.exists(dir):
		idx += 1
		dir = os.path.join(args.save_dir, algo, str(idx))
	os.makedirs(dir)
	with open(os.path.join(dir, "args.json"), "w") as f:
		json.dump(args.__dict__, f, indent=2)
	writer = SummaryWriter()
	eval.evaluate(model)
	
	if args.plot:
		Reward_log = []
		step_interval = args.plot_interval

	steps = 0

	for epoch in trange(args.n_epochs):
		with tqdm(total = len(dataLoader)) as pbar:
			for batch in dataLoader:
				# Get data
				# print(batch.keys())
				state = batch['state'].float().to(args.device)
				action = batch['action'].float().to(args.device)
				next_states = batch['next_state'].float().to(args.device)
				reward = batch['reward'].float().to(args.device)
				not_done = batch['not_done'].float().to(args.device)
				select = batch['select'].float().to(args.device)

				Recon_loss, KL_loss, Critic_loss, Actor_loss = model.train(state, action, next_states, reward, not_done, select)
				
				pbar.set_description("Epoch: {}".format(epoch))
				# pbar.set_postfix(VAE_loss=Recon_loss+0.5 * KL_loss, Critic_loss = Critic_loss, distrub_loss = Actor_loss)
				pbar.set_postfix(VAE_loss=Recon_loss+0.5 * KL_loss, Critic_loss = Critic_loss, distrub_loss = Actor_loss, vae_lr = model.VAE_optim.param_groups[0]['lr'], actor_lr = model.Actor_disturb_optim.param_groups[0]['lr'], critic_lr = model.Critic_optim.param_groups[0]['lr'])
				pbar.update(1)
				
				if args.plot and ((steps+len(state)) // step_interval)-steps//step_interval>0:
					Reward, _ = eval.evaluate(model)
					Reward_log.append(Reward)
					writer.add_scalar("Reward", Reward, steps)
				steps += len(state)

				writer.add_scalar("Recon_loss", Recon_loss, steps)
				writer.add_scalar("KL_loss", KL_loss, steps)
				writer.add_scalar("Critic_loss", Critic_loss, steps)
				writer.add_scalar("Actor_loss", Actor_loss, steps)
		
		Reward, episodes_len = eval.evaluate(model)
		if Reward> Mx_Reward:
			torch.save(model.state_dict(), os.path.join(dir, algo+"_best.pth"))
			Mx_Reward = Reward
		torch.save(model.state_dict(), os.path.join(dir, algo+f"_{epoch%10}.pth"))
		print("Epoch: {}, Reward: {}, Mean Episodes Length: {}".format(epoch, Reward, episodes_len))
		print("####################################")

	if args.plot:	
		Reward_log = np.array(Reward_log)
		np.save(os.path.join(dir, algo+"_reward.npy"), Reward_log)
		
		return Reward_log.tolist()


if __name__ == "__main__":
	args = get_args("babcq")

	replay_buffer = utils.ReplayBuffer()
	replay_buffer.load(args.dataset_path, args.file_name, args.trajectory_truncation)

	states, returns = get_mcret(replay_buffer, args)

	ue_model, C = train_ue(states, returns, args)

	selected_buffer, selected_len, border = select_batch_ue(replay_buffer, states, returns, ue_model, C, args)

	dataset = SamaplesDataset.from_buffer(selected_buffer)
	dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
	

	if args.plot:
		step_interval = args.plot_interval
		Reward_logs = []

		for _ in range(args.training_iteration):
			model = BCQ(device = device, gamma = args.gamma, latent_dim = args.latent_dim, lr = args.lr, lr_critic = args.lr_critic).to(device)
			Reward_log = train(model, dataloader, args, "BABCQ")
			Reward_logs.append(Reward_log)
		
		Reward_logs = np.array(Reward_logs)
		np.save(os.path.join(args.save_dir, "BABCQ_Rewards.npy"), Reward_logs)
		plot_eval(step_interval, Reward_logs, "BABCQ")
	
