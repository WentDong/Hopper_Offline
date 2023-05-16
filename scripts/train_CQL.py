import torch
import numpy as np

import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))


from evaluate import evaluation
from dataloader import SamaplesDataset
from args import get_args
from agents.CQL.cql_agent import CQL
from agents.CQL.cql_train import cql_train
from torch.utils.data import DataLoader
from tqdm import *


def train(dataLoader, args):
	cql_train(dataLoader, args)

if __name__ == "__main__":
	args = get_args()
	dataset = SamaplesDataset('../'+ args.dataset_path, args.file_name)
	dataLoader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle = True
	)
	train(dataLoader, args)