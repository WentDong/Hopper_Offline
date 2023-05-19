import os
import numpy as np
import copy

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

	def sample(self, batch_size, require_idxs=False, space_rollout=0):
		ind = np.random.randint(0, len(self.storage) - space_rollout,
								size=batch_size)
		if require_idxs:
			return np.array(self.storage[ind]), ind
		else:
			return np.array(self.storage[ind])			
			

		

def Mount_Carlo_Estimation(Replay_Buffer: Traj_Replay_Buffer, space_rollout=0, Sample_Size = -1, discount_factor = 0.99):
	if Sample_Size == -1:
		Trajs = np.array(Replay_Buffer.storage.copy())
	else:
		Trajs = Replay_Buffer.sample(Sample_Size, space_rollout=space_rollout)
	rewards = [Trajs[i][3] for i in range(len(Trajs))]  # List containing the rewards for each sample.
	monte_carlo_returns = []  # List containing the Monte-Carlo returns.
	monte_carlo_return = 0
	t = 0  # Exponent by which the discount factor is raised.
	i = 0
	while i < len(Trajs):
		
		while not Trajs[i][4]:  # Execute until you encounter a terminal state.

			# Equation to calculate the Monte-Carlo return.
			monte_carlo_return += discount_factor ** t * rewards[i]
			i += 1  # Go to the next sample.
			t += 1  # Increasing the exponent by which the discount factor is raised.

			# Condition to check whether we have reached the end of the replay memory without the episode being
			# terminated, and if so break. (This can happen with the samples at the end of the replay memory as we
			# only store the samples till we reach the replay memory size and not till we exceed it with the episode
			# being terminated.)
			if i == len(Trajs):

				# If the episode hasn't terminated but you reach the end append the Monte-Carlo return to the list.
				monte_carlo_returns.append(monte_carlo_return)

				# Resetting the Monte-Carlo return value and the exponent to 0.
				monte_carlo_return = 0
				t = 0

				break  # Break from the loop.

		# If for one of the samples towards the end we reach the end of the replay memory and it hasn't terminated,
		# we will go back to the beginning of the for loop to calculate the Monte-Carlo return for the future
		# samples if any for whom the episode hasn't terminated.
		if i == len(Trajs):
			continue

		# Equation to calculate the Monte-Carlo return.
		monte_carlo_return += discount_factor ** t * rewards[i]

		# Appending the Monte-Carlo Return for cases where the episode terminates without reaching the end of the
		# replay memory.
		monte_carlo_returns.append(monte_carlo_return)

		# Resetting the Monte-Carlo return value and the exponent to 0.
		monte_carlo_return = 0
		t = 0

	monte_carlo_returns = np.array(monte_carlo_returns)

	# Normalizing the returns. 
	# monte_carlo_returns = (monte_carlo_returns - np.mean(monte_carlo_returns)) / (np.std(monte_carlo_returns)
	# 																				+ 1e-08)
	# monte_carlo_returns = monte_carlo_returns.tolist()

	return monte_carlo_returns

