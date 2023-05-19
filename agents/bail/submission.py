import argparse
import os
from pathlib import Path
import sys
import torch
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from bail_agent import BAIL

agent = BAIL(state_dim=11, action_dim=3, hidden_dim=512)
BC_net = os.path.dirname(os.path.abspath(__file__)) + "/BAIL_best.pth"
state_dict=  torch.load(BC_net)
agent.load_state_dict(state_dict)
def my_controller(observation, action_space, is_act_continuous=True):
    action = agent.take_action(observation['obs'])
    return [action]