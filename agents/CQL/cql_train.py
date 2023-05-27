import numpy as np
import inspect
import os
import json
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))
from tqdm import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.CQL.cql_agent import CQL
from scripts.evaluate import Evaluator
from scripts.utils import plot_eval

# train & evaluate
def cql_train(dataLoader, args):
    '''设置参数'''
    # batch_num = len(dataLoader)
    beta = args.cql_beta
    num_random = args.cql_random_num    # the original try is 5
    # num_trains_per_train_loop = len(dataLoader)
    actor_lr = args.cql_alr
    critic_lr = args.cql_clr
    alpha_lr = args.cql_alr
    hidden_dim = args.cql_hidden_dim
    gamma = args.gamma
    tau = args.cql_tau  # 软更新参数
    action_bound = args.cql_ac_bound
    n_epochs = args.cql_n_epochs
    '''****************hyper-parameters*****************'''
    Reward_log = []
    steps = 0
    step_interval = 64000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_entropy = -np.prod(3).item()
    Eval = Evaluator(device = device)

    agent = CQL(state_dim=11, action_dim=3, hidden_dim=hidden_dim, action_bound = action_bound, actor_lr = actor_lr,
            critic_lr = critic_lr, alpha_lr = alpha_lr, target_entropy = target_entropy, tau = tau,
            gamma = gamma, device = device, beta=beta, num_random=num_random)

    max_reward = 0
    for epoch in trange(n_epochs):
        with tqdm(total = len(dataLoader)) as pbar:
            for batch in dataLoader:
			    # data = [batch['state'], batch['action'], batch['reward'], batch['next_state'], 1-batch['not_done']]
                transition_dict = {'states':batch['state'], 'actions':batch['action'], 
		       						'rewards':batch['reward'], 'next_states':batch['next_state'], 'dones':1-batch['not_done']}
                agent.update(transition_dict)
                pbar.update(1)

                steps += args.batch_size
                if steps % step_interval == 0:
                    Reward, episode_len = Eval.evaluate(agent.actor)
                    Reward_log.append(Reward)
        
        Reward, episode_len = Eval.evaluate(agent.actor)
         
        
        # evaluate and save actor model
        actor = agent.actor
        critic_1 = agent.critic_1; critic_2 = agent.critic_2
        target_critic_1 = agent.target_critic_1; target_critic_2 = agent.target_critic_2
        log_alpha = agent.log_alpha

        rewards, total_steps = Eval.evaluate(actor)
        print('Epoch:{}, rewards:{}, total_steps:{}'.format(epoch, rewards, total_steps))
        if rewards > max_reward:
            print("save new model")
            max_reward = rewards
            torch.save(actor.state_dict(), os.path.join(currentdir, "actor.pth"))
            torch.save(critic_1.state_dict(), os.path.join(currentdir, "critic_1.pth"))
            torch.save(critic_2.state_dict(), os.path.join(currentdir, "critic_2.pth"))
            torch.save(target_critic_1.state_dict(), os.path.join(currentdir, "target_critic_1.pth"))
            torch.save(target_critic_2.state_dict(), os.path.join(currentdir, "target_critic_2.pth"))
            torch.save(log_alpha, os.path.join(currentdir, "log_alpha.pth"))

    return Reward_log
