import torch
import numpy as np

import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))


from scripts.evaluate import Evaluator
from scripts.dataloader import SamaplesDataset, TrajectoryDataset
from agents.CQL.cql_agent import CQL
from agents.CQL.cql_train import cql_train
from torch.utils.data import DataLoader

def cql_dataset(dataset, if_clip = False, truncation = 0.1):
    if not if_clip:
        return dataset
    traj = dataset
    single_traj = []
    new_dataset = []
    lenth = 0
    for i in range(len(traj)):
        transition = traj[i]
        if transition['not_done'] == 0:
            lenth += 1
            if truncation > 0:
                lenth = int(lenth * (1 - truncation))
            
            for j in range(lenth):
                new_dataset.append(single_traj[j])
            single_traj = []
            lenth = 0
        else:
            lenth += 1
            single_traj.append(traj[i])
    return new_dataset
    