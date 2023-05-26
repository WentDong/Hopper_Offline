import numpy as np
import torch
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
from agents.BCQ.bcq_agent import BCQ
from torch.utils.data import DataLoader
from tqdm import *
from torch.utils.tensorboard import SummaryWriter
from train_BAIL import get_mcret, train_ue, select_batch_ue
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from train_BCQ import train

if __name__ == "__main__":
	args = get_args("bail")

	replay_buffer = utils.ReplayBuffer()
	replay_buffer.load(args.dataset_path, args.file_name, args.trajectory_truncation)

	states, returns = get_mcret(replay_buffer, args)

	ue_model, C = train_ue(states, returns, args)

	selected_buffer, selected_len, border = select_batch_ue(replay_buffer, states, returns, ue_model, C, args)

	dataset = SamaplesDataset.from_buffer(selected_buffer)
	dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
	
	model = BCQ(device = device, lr = args.lr).to(device)
	train(model, dataloader, args, "BABCQ")