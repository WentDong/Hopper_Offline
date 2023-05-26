import argparse
import os
from pathlib import Path
import sys
import torch
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from bcq_agent import BCQ

agent = BCQ(latent_dim=10)
BCQ_net = os.path.dirname(os.path.abspath(__file__)) + "/BCQ_best.pth"
state_dict=  torch.load(BCQ_net)
agent.load_state_dict(state_dict)
def my_controller(observation, action_space, is_act_continuous=True):
    action = agent.take_action(observation['obs'])
    return [action]