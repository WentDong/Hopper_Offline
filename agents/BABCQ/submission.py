import argparse
import os
import json
from pathlib import Path
import sys
import torch
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from bcq_agent import BCQ

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "args.json"), "r") as f:
    args = argparse.Namespace(**json.load(f))
agent = BCQ(gamma = args.gamma, latent_dim = args.latent_dim)
BCQ_net = os.path.dirname(os.path.abspath(__file__)) + "/BABCQ_best.pth"
state_dict=  torch.load(BCQ_net, map_location=torch.device('cpu'))
agent.load_state_dict(state_dict)
def my_controller(observation, action_space, is_act_continuous=True):
    action = agent.take_action(observation['obs'])
    return [action]