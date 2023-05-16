import numpy as np
import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))
from tqdm import *
import gym
from env.chooseenv import make

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.CQL.cql_agent import CQL
from scripts.evaluate import evaluation

# train & evaluate
def cql_train(dataLoader, args):
    '''设置参数'''
    batch_num = len(dataLoader)
    beta = 5.0
    num_random = 5
    num_trains_per_train_loop = len(dataLoader)
    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.99
    tau = 0.005  # 软更新参数
    batch_size = 64
    action_bound = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_entropy = -np.prod(3).item()


    agent = CQL(state_dim=11, action_dim=3, hidden_dim=128, action_bound = action_bound, actor_lr = actor_lr,
            critic_lr = critic_lr, alpha_lr = alpha_lr, target_entropy = target_entropy, tau = tau,
            gamma = gamma, device = device, beta=beta, num_random=num_random)

    max_reward = 0
    for epoch in trange(args.n_epochs):
        with tqdm(total = len(dataLoader)) as pbar:
            for batch in dataLoader:
			    # data = [batch['state'], batch['action'], batch['reward'], batch['next_state'], 1-batch['not_done']]
                transition_dict = {'states':batch['state'], 'actions':batch['action'], 
		       						'rewards':batch['reward'], 'next_states':batch['next_state'], 'dones':1-batch['not_done']}
                agent.update(transition_dict)
                pbar.update(1)
        
        # evaluate and save actor model
        actor = agent.actor
        rewards, total_steps = evaluation(actor)
        print('Epoch:{}, rewards:{}, total_steps:{}'.format(epoch, rewards, total_steps))
        if rewards > max_reward:
            print("save new model")
            max_reward = rewards
            torch.save(actor.state_dict(), os.path.join(parentdir, "CQL", "actor.pth"))






















# def cql_train(model, dataLoader, args):
#     # env_name = ['classic_Acrobot-v1', 'gym_Hopper-v2']
#     env_name = 'gym_Hopper-v2'
#     env = make(env_name)
#     print(env)

#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     Mx_Reward = 0
#     dir = os.path.join(args.save_dir, "CQL")
#     if not os.path.exists(dir):
#         os.makedirs(dir)
#     for epoch in trange(args.n_epochs):
#         with tqdm(total = len(dataLoader)) as pbar:
#             for batch in dataLoader:
#                 print(batch.keys())
#                 # Get data
#                 transition_dict = {'states':batch['state'],'actions':batch['action'],
#                                    'next_states':batch['next_state'], 'rewards':batch['reward'],
#                                    'dones':batch['not_done']}
#                 model.update(transition_dict)
#                 # state = batch['state'].float().to(args.device)
#                 # # print(state.shape, type(state))
#                 # action = batch['action'].float().to(args.device)
#                 # loss = model.train(state, action)
                
#                 # optimizer.zero_grad()
#                 # loss.backward()
#                 # # Update parameters
#                 # optimizer.step()
#                 pbar.set_description("Epoch: {}".format(epoch))
#                 pbar.set_postfix(loss=loss.item())
#                 pbar.update(1)
#                 # Print loss
#         Reward = cql_evaluation(model)
#         if Reward> Mx_Reward:
#             torch.save(model, os.path.join(dir, "CQL_best.pth"))

#         torch.save(model, os.path.join(dir,"BC_{}.pth".format(epoch%10)))
#         # tqdm.set_description("Epoch: {}, Reward: {}".format(epoch, Reward))
#         print("Epoch: {}, Reward: {}".format(epoch, Reward))

# def cql_evaluation(model):
#     pass