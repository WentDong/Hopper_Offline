'''
USE a Simple MLP to predict the action from state
Loss: MSE loss for action
'''
import torch
class BC(torch.nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim=128):
		super(BC, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.fc1 = torch.nn.Linear(self.state_dim, hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = torch.nn.Linear(hidden_dim, self.action_dim)
		self.relu = torch.nn.ReLU()
		self.tanh = torch.nn.Tanh()
		self.loss = torch.nn.MSELoss()
		
	def forward(self, state):
		x = self.fc1(state)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc3(x)
		x = self.tanh(x)
		return x
	
	def train(self, state, action):
		pred = self.forward(state)
		loss = self.loss(pred, action)
		return loss
	
	def take_action(self, state):
		# print(state)
		state = torch.tensor(state, device=next(self.parameters()).device).float()
		action = self.forward(state)
		# print(action.detach().numpy())
		return action.detach().cpu().numpy()


