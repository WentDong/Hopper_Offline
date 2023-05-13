import torch
import numpy as np
import gym
def evaluation(model):
	env = gym.make('Hopper-v2')
	state = env.reset()
	env.max_episodes_length = 1000
	env.num_envs = 100
	Accumulate_Reward = np.zeros(env.num_envs)
	dones = np.zeros(env.num_envs)
	total_steps = 0

	for _ in range(env.max_episodes_length):
		total_steps += np.sum(1-dones)
		action = model.take_action(state)
		state, reward, done, _ = env.step(action)
		Accumulate_Reward += reward * (1-dones)
		dones += done
		if np.sum(dones) == env.num_envs:
			break
	return Accumulate_Reward.mean(), total_steps/env.num_envs