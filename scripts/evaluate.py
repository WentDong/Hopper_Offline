import torch
import numpy as np
import gym
import copy
import time
from torch.multiprocessing import Pool, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass


def eval(env, model, max_episodes_length = 1000):
	state = env.reset()
	Accumulate_Reward = 0
	total_steps = 0
	for _ in range(max_episodes_length):
		total_steps += 1
		action = model.take_action(state)
		state, reward, done, _ = env.step(action)
		Accumulate_Reward += reward
		if done:
			break
	return Accumulate_Reward, total_steps

def Parallel_model(model, len = 20, device = 'cpu'):
	model_cpu = copy.deepcopy(model).to(device)
	model_list = [model_cpu for _ in range(len)]
	return model_list

def Parallel_env(env = 'Hopper-v2', len = 20):
	env_list = [gym.make(env) for _ in range(len)]
	return env_list
	

def evaluation(model, env = 'Hopper-v2', len = 20, device = 'cpu'):
	if device == 'cpu':
		model_list = Parallel_model(model, len, device = device)
		env_list = Parallel_env(env, len)
		Zip = zip(env_list, model_list)
		pool = Pool(len(env_list))
		result = pool.starmap(eval, Zip)
		Accumulate_Reward, total_steps = zip(*result)

        # print(np.array(Accumulate_Reward).mean(), np.array(total_steps).mean())
        # input()
		return np.array(Accumulate_Reward).mean(), np.array(total_steps).mean()
	
	else:
		env = gym.make(env)
		env.max_episodes_length = 1000
		Accumulate_Reward = 0
		total_steps = 0
        
		num_envs = 20
		for i in range(num_envs):
			state = env.reset()
			for _ in range(env.max_episodes_length):
				total_steps += 1
				action = model.take_action(state)
				state, reward, done, _ = env.step(action)
				Accumulate_Reward += reward
				if done:
					break
            
		return Accumulate_Reward/num_envs, total_steps/num_envs
