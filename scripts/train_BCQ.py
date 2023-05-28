import torch

import json
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
from utils import plot_eval
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
	if args.plot:
		step_interval = args.plot_interval
		Reward_log = []
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
				Recon_loss, KL_loss, Critic_loss, Actor_loss = model.train(state, action, next_states, reward, not_done)
				
				pbar.set_description("Epoch: {}".format(epoch))
				pbar.set_postfix(VAE_loss=Recon_loss+0.5 * KL_loss, Critic_loss = Critic_loss, distrub_loss = Actor_loss)
				pbar.update(1)

				writer.add_scalar("Recon_loss", Recon_loss, steps)
				writer.add_scalar("KL_loss", KL_loss, steps)
				writer.add_scalar("Critic_loss", Critic_loss, steps)
				writer.add_scalar("Actor_loss", Actor_loss, steps)
				if args.plot and ((steps+len(state)) // step_interval)-steps//step_interval>0:
					Reward, _ = eval.evaluate(model)
					Reward_log.append(Reward)
				steps += len(state)

		
		Reward, episodes_len = eval.evaluate(model)
		if Reward> Mx_Reward:
			torch.save(model.state_dict(), os.path.join(dir, algo+"_best.pth"))
			Mx_Reward = Reward
		torch.save(model.state_dict(), os.path.join(dir, algo+f"_{epoch%10}.pth"))
		print("Epoch: {}, Reward: {}, Mean Episodes Length: {}".format(epoch, Reward, episodes_len))
		print("####################################")
		with open(os.path.join(dir, "log.txt"), "a") as f:
			f.write("Epoch: {}, Reward: {}, Mean Episodes Length: {}\n".format(epoch, Reward, episodes_len))
	if args.plot:
		return Reward_log
if __name__ == "__main__":
	args = get_args("bcq")
	dataset = TrajectoryDataset(args.dataset_path, args.file_name, args.trajectory_truncation)
	dataset = SamaplesDataset.from_traj(dataset)
	dataLoader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle = True
	)
	if not args.plot:
		model = BCQ(device = args.device, gamma = args.gamma, latent_dim = args.latent_dim, lr = args.lr, lr_critic = args.lr_critic).to(args.device)
		train(model, dataLoader, args)
	else:
		step_interval = args.plot_interval
		Reward_logs = []
		for _ in range(args.training_iteration):
			model = BCQ(device = args.device, gamma = args.gamma, latent_dim = args.latent_dim, lr = args.lr, lr_critic = args.lr_critic).to(args.device)
			Reward_log = train(model, dataLoader, args)
			Reward_logs.append(Reward_log)
		Reward_logs = np.array(Reward_logs)
		np.save(os.path.join(args.save_dir, "BCQ_Reward_logs.npy"), Reward_logs)
		plot_eval(step_interval, Reward_logs, "BCQ")
