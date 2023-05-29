import torch
import numpy as np

import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))


from evaluate import Evaluator
from dataloader import SamaplesDataset, TrajectoryDataset
from args import get_args
from agents.CQL.cql_agent import CQL
from agents.CQL.cql_train import cql_train
from agents.CQL.cql_dataset import cql_dataset
from torch.utils.data import DataLoader
from scripts.utils import plot_eval


def train(dataLoader, args):
	if args.plot:
		Reward_logs = []
		for _ in range(args.training_iteration):
			Reward_log = cql_train(dataLoader, args)
			Reward_logs.append(Reward_log)
		return Reward_logs
	else:
		cql_train(dataLoader, args)
if __name__ == "__main__":
	args = get_args("cql")
	dataset = SamaplesDataset(args.dataset_path, args.file_name)
	dataset = cql_dataset(dataset,if_clip = True, truncation = 0.1)

	dataLoader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle = True,
	)
	if args.plot:
		step_interval = args.plot_interval
		Reward_logs = train(dataLoader, args)
		Reward_logs = np.array(Reward_logs)
		np.save(os.path.join(args.save_dir, "CQL_Reward_logs.npy"), Reward_logs)
		plot_eval(step_interval, Reward_logs, "CQL")    # I only use batch_size for plot_eval
	else:
		train(dataLoader, args)