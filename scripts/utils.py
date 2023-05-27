import os
import numpy as np
import copy
import torch
# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# https://github.com/aviralkumar2907/BEAR/blob/master/utils.py

'''A standard replay buffer'''

class Sample_ReplayBuffer(object):
	def __init__(self):

		self.keys = ["state", "next_state", "action", "reward", "done"]

		self.storage = []

		self.with_mask = False


	# Expects tuples of (state, next_state, action, reward, done)
	def add(self, data):
		self.storage.append(data)

	def sample(self, batch_size, require_idxs=False, space_rollout=0):
		ind = np.random.randint(0, len(self.storage) - space_rollout,
								size=batch_size)
		state, next_state, action, reward, done = [], [], [], [], []

		for i in ind: 
			s, s2, a, r, d = self.storage[i]
			state.append(np.array(s, copy=False))
			next_state.append(np.array(s2, copy=False))
			action.append(np.array(a, copy=False))
			reward.append(np.array(r, copy=False))
			done.append(np.array(d, copy=False))

		if require_idxs:
			return (np.array(state),
					np.array(next_state),
					np.array(action),
					np.array(reward).reshape(-1, 1),
					np.array(done).reshape(-1, 1), ind)
		elif self.with_mask:
			mask = self.bootstrap_mask[ind]
			return (np.array(state),
					np.array(next_state),
					np.array(action),
					np.array(reward).reshape(-1, 1),
					np.array(done).reshape(-1, 1),
					np.array(mask))
		else:
			return (np.array(state),
					np.array(next_state),
					np.array(action),
					np.array(reward).reshape(-1, 1),
					np.array(done).reshape(-1, 1))

	def index(self, i):
		return self.storage[i]

	# def save(self, filename):
	# 	np.save("./buffers/"+filename+"sars.npy", self.storage)

	def load(self, path, filename, truncation, bootstrap_dim=None):
		self.name = filename + f"_trunc{truncation}" if truncation > 0 else filename
		start = 0
		data = {}
		keys = ["state", "next_state", "action", "reward", "not_done"]
		for key in keys:
			data[key] = np.load(os.path.join(path, filename + "_" + key + ".npy"))
		for i in range(len(data['state'])):
			self.add((data['state'][i], data['next_state'][i], data['action'][i], data['reward'][i], 1 - data['not_done'][i]))
			if truncation > 0:
				if data['not_done'][i] == 0:
					length = i - start + 1
					start = i + 1
					self.storage = self.storage[:-int(length * truncation)]
					self.storage[-1] = (self.storage[-1][0], self.storage[-1][1], self.storage[-1][2], self.storage[-1][3], 1)



		num_samples = len(self.storage)
		print('Load buffer size:', num_samples)
		if bootstrap_dim is not None:
			print('Bootstrap with dim', bootstrap_dim)
			self.with_mask = True
			self.bootstrap_dim = bootstrap_dim
			bootstrap_mask = np.random.binomial(n=1, size=(1, num_samples, bootstrap_dim,), p=0.8)
			bootstrap_mask = np.squeeze(bootstrap_mask, axis=0)
			self.bootstrap_mask = bootstrap_mask[:num_samples]

	def cut_final(self, buffer_size):
		self.storage = self.storage[ -int(buffer_size): ]

	def get_length(self):
		return self.storage.__len__()

class Traj_Replay_Buffer(object):
	def __init__(self):

		self.keys = ["state", "next_state", "action", "reward", "done"]

		self.storage = []

		self.with_mask = False

	def add(self, data):
		self.storage.append(data)
	def load(self, Traj_dataset):
		for Traj in Traj_dataset.Trajectories:
			self.add(Traj)
	def sample(self, batch_size, require_idxs=False, space_rollout=0):
		ind = np.random.randint(0, len(self.storage) - space_rollout,
								size=batch_size)
		# print(len(self.storage))
		# print(ind)
		# print(self.storage[ind])
		ret = [self.storage[i] for i in ind]
		# print(len(ret))
		# print(len(ret[0]), ret[0][0])
		if require_idxs:
			return ret, ind
		else:
			return ret		
			

		

def Mount_Carlo_Estimation(Replay_Buffer: Traj_Replay_Buffer, space_rollout=0, Sample_Size = -1, discount_factor = 0.95):
	if Sample_Size == -1:
		Trajs = np.array(Replay_Buffer.storage)
	else:
		Trajs = Replay_Buffer.sample(Sample_Size, space_rollout=space_rollout)
	
	Batch_monte_carlo_returns = []

	for i in range(len(Trajs)):
		Traj = Trajs[i]

		rewards = [Traj[j][3][0] for j in range(len(Traj))]  # List containing the rewards for each sample.
		
		monte_carlo_returns = []  # List containing the Monte-Carlo returns.
		monte_carlo_return = 0

		for j in range(len(Traj)-1, -1, -1):
			# print(j, len(Traj), len(rewards))
			monte_carlo_return = monte_carlo_return * discount_factor + rewards[j]
			monte_carlo_returns = [monte_carlo_return] + monte_carlo_returns
		# print(len(monte_carlo_returns))
		Batch_monte_carlo_returns.append(np.array(monte_carlo_returns))
		# print(len(Batch_monte_carlo_returns), Batch_monte_carlo_returns[-1].shape)
	

	# Normalizing the returns. 
	# monte_carlo_returns = (monte_carlo_returns - np.mean(monte_carlo_returns)) / (np.std(monte_carlo_returns)
	# 																				+ 1e-08)
	# monte_carlo_returns = monte_carlo_returns.tolist()

	return Batch_monte_carlo_returns, Trajs

def compute_advantage(gamma, lmbda, td_delta):
	td_delta = td_delta.detach().numpy()
	advantage_list = []
	advantage = 0.0
	for delta in td_delta[::-1]:
		advantage = gamma * lmbda * advantage + delta
		advantage_list.append(advantage)
	advantage_list.reverse()
	return torch.tensor(advantage_list, dtype=torch.float)


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd

def plot_eval(step_interval, rewards, runs, dir = "figs"):
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111)
	# df_reward = pd.DataFrame(train_recorded_episode_reward_log).melt()
	rewards = np.array(rewards)
	df_reward  = pd.DataFrame(rewards).melt()
	sns.lineplot(ax = ax, x='variable', y="value", data=df_reward, ci="sd", label="Reward")
	# sns.lineplot(ax=ax, x=steps, y = rewards)
	ax.set_title(f"Train learning Curve for {runs} runs")
	ax.set_ylabel("Episodic Reward")
	ax.set_xlabel("Steps * " + str(step_interval))
	ax.legend(loc="lower right")
	plt.tight_layout()
	if not os.path.exists(dir):
		os.makedirs(dir)
	plt.savefig(os.path.join(dir, runs+"train_learning_curve.png"))
	plt.show()

