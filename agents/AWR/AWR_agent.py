import torch
from torch import nn

class AWR(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim = 256):
		super(AWR, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.beta = 0.05
		self.lamb = 0.9
		self.gamma = 0.95
		self.actor = nn.Sequential(
			nn.Linear(self.state_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim//2),
			nn.ReLU(),
			nn.Linear(hidden_dim//2, self.action_dim),
			nn.Tanh()
		)
		self.critic = nn.Sequential(
			nn.Linear(self.state_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim//2),
			nn.ReLU(),
			nn.Linear(hidden_dim//2, 1)
		)

	def forward(self, state):
		return self.actor(state)
	
	def take_action(self, state):
		state = torch.tensor(state).float()
		action = self.forward(state)
		return action.detach().numpy()