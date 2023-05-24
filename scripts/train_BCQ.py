import torch

import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))

from agents.BCQ.bcq_agent import BCQ
from args import get_args
from dataloader import *
from torch.utils.data import DataLoader
from tqdm import *
from evaluate import *
from torch.utils.tensorboard import SummaryWriter

def train(model, dataLoader, args):
	eval = Evaluator(device =  args.device)
	Mx_Reward = 0
	idx =  0
	steps = 0
	dir = os.path.join(args.save_dir, "BCQ", str(idx))
	while os.path.exists(dir):
		idx += 1
		dir = os.path.join(args.save_dir, "BCQ", str(idx))
	os.makedirs(dir)
	with open(os.path.join(dir, "args.txt"), "w") as f:
		f.write(str(args))
	writer = SummaryWriter()
	eval.evaluate(model)
	for epoch in trange(args.n_epochs):
		with tqdm(total = len(dataLoader)) as pbar:
			for batch in dataLoader:
				# Get data
				state = batch['state'].float().to(args.device)

				action = batch['action'].float().to(args.device)
				next_states = batch['next_state'].float().to(args.device)
				reward = batch['reward'].float().to(args.device)
				not_done = batch['not_done'].float().to(args.device)
				Recon_loss, KL_loss, Critic_loss, Actor_loss = model.train(state, action, next_states, reward, not_done)
				
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
			torch.save(model.state_dict(), os.path.join(dir, "BCQ_best.pth"))
			Mx_Reward = Reward
		torch.save(model.state_dict(), os.path.join(dir, f"BCQ_{epoch%10}.pth"))
		print("Epoch: {}, Reward: {}, Mean Episodes Length: {}".format(epoch, Reward, episodes_len))

if __name__ == "__main__":
	args = get_args()
	model = BCQ(device = args.device).to(args.device)
	dataset = TrajectoryDataset(args.dataset_path, args.file_name, args.trajectory_truncation)
	dataset = SamaplesDataset.from_traj(dataset)
	dataLoader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle = True
	)
	train(model, dataLoader, args)
