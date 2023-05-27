import argparse
import os
from pathlib import Path
import sys
import torch
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from cql_agent import CQL
import numpy as np

'''设置submission参数'''
# batch_num = len(dataLoader)
beta = 5.0
num_random = 5    # the original try is 5
actor_lr = 3e-4
critic_lr = 3e-3
alpha_lr = 3e-4
num_episodes = 100
hidden_dim = 128
gamma = 0.99
tau = 0.005  # 软更新参数
batch_size = 64
action_bound = 1
n_epochs = 80
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_entropy = -np.prod(3).item()
'''****************hyper-parameters*****************'''
def my_load_model(agent):
    print(os.path.join(base_dir, "CQL", "actor.pth"))
    agent.actor.load_state_dict(torch.load(os.path.join(base_dir, "actor.pth")))
    agent.critic_1.load_state_dict(torch.load(os.path.join(base_dir, "critic_1.pth")))
    agent.critic_2.load_state_dict(torch.load(os.path.join(base_dir, "critic_2.pth")))
    agent.log_alpha = torch.load(os.path.join(base_dir, "log_alpha.pth"))
    agent.target_critic_1.load_state_dict(torch.load(os.path.join(base_dir, "target_critic_1.pth")))
    agent.target_critic_2.load_state_dict(torch.load(os.path.join(base_dir, "target_critic_2.pth")))


agent = CQL(state_dim=11, action_dim=3, hidden_dim=128, action_bound = action_bound, actor_lr = actor_lr,
        critic_lr = critic_lr, alpha_lr = alpha_lr, target_entropy = target_entropy, tau = tau,
        gamma = gamma, device = device, beta=beta, num_random=num_random)
my_load_model(agent)

def my_controller(observation, action_space, is_act_continuous=True):
    action = agent.take_action(observation['obs'])
    return [action]

