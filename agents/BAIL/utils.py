import os.path

import numpy as np

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# https://github.com/aviralkumar2907/BEAR/blob/master/utils.py

'''A standard replay buffer'''

class ReplayBuffer(object):
	def __init__(self, countdown=False):

		self.countdown = countdown
		
		self.keys = ["state", "next_state", "action", "reward", "done"]
		if countdown:
			self.keys.append("countdown")

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

	def index(self, i, full=False):
		if full:
			return self.storage[i]
		else:
			return self.storage[i][:5]

	# def save(self, filename):
	# 	np.save("./buffers/"+filename+"sars.npy", self.storage)

	def load(self, path, filename, truncation, bootstrap_dim=None):
		self.name = filename + f"_trunc{truncation}" if truncation > 0 else filename
		start = 0
		data = {}
		keys = ["action", "next_state", "not_done", "reward", "state"]
		for key in keys:
			data[key] = np.load(os.path.join(path, filename + "_" + key + ".npy"))
		for i in range(len(data['state'])):
			self.add((data['state'][i], data['next_state'][i], data['action'][i], data['reward'][i], 1 - data['not_done'][i]))
			if data['not_done'][i] == 0:
				length = i - start + 1
				if self.countdown:
					for t, j in enumerate(range(start, i + 1)):
						self.storage[j] = (*self.storage[j], np.array([length - t]))
						assert length - t > 0
				if truncation > 0:
					self.storage = self.storage[:-int(length * truncation)]
					self.storage[-1] = (self.storage[-1][0], self.storage[-1][1], self.storage[-1][2], self.storage[-1][3], 1)
				start = i + 1
		# deal with last episode
		if self.countdown:
			self.storage = self.storage[:start]

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

