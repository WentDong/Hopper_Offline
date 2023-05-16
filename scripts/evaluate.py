import torch
import numpy as np
import gym
def evaluation(model):
	env = gym.make('Hopper-v2')
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