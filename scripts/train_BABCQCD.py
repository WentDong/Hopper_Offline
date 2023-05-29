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
from agents.bail import utils
from agents.bail.mcret import *
from agents.bail.bail_training import Value, train_upper_envelope
from agents.BCQCD.bcqcd_agent import BCQCD
from torch.utils.data import DataLoader
from tqdm import *
from torch.utils.tensorboard import SummaryWriter
from train_BAILCD import get_mcret, train_ue, select_batch_ue
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
				countdown = batch['countdown'].float().to(args.device)

				Recon_loss, KL_loss, Critic_loss, Actor_loss = model.train(state, action, next_states, reward, not_done, select, countdown)
				
				pbar.set_description("Epoch: {}".format(epoch))
				pbar.set_postfix(VAE_loss=Recon_loss+0.5 * KL_loss, Critic_loss = Critic_loss, distrub_loss = Actor_loss)
				pbar.update(1)

				writer.add_scalar("Recon_loss", Recon_loss, steps)
				writer.add_scalar("KL_loss", KL_loss, steps)
				writer.add_scalar("Critic_loss", Critic_loss, steps)
				writer.add_scalar("Actor_loss", Actor_loss, steps)
				steps += 1
		
		Reward, episodes_len = eval.evaluate(model)
		if Reward> Mx_Reward:
			torch.save(model.state_dict(), os.path.join(dir, algo+"_best.pth"))
			Mx_Reward = Reward
		torch.save(model.state_dict(), os.path.join(dir, algo+f"_{epoch%10}.pth"))
		print("Epoch: {}, Reward: {}, Mean Episodes Length: {}".format(epoch, Reward, episodes_len))
		print("####################################")


if __name__ == "__main__":
	args = get_args("babcqcd")

	replay_buffer = utils.ReplayBuffer(countdown=True)
	replay_buffer.load(args.dataset_path, args.file_name, args.trajectory_truncation)

	states, returns = get_mcret(replay_buffer, args)

	ue_model, C = train_ue(states, returns, args)

	selected_buffer, selected_len, border = select_batch_ue(replay_buffer, states, returns, ue_model, C, args)

	dataset = SamaplesDataset.from_buffer(selected_buffer)
	dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
	
	model = BCQCD(device = device, rollout = args.rollout, gamma = args.gamma, latent_dim = args.latent_dim, lr = args.lr, lr_critic = args.lr_critic).to(device)
	train(model, dataloader, args, "BABCQCD")