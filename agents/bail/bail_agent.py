'''
USE a Simple MLP to predict the action from state
Loss: MSE loss for action
'''
import torch


class Value(torch.nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='relu', init_small_weights=False, init_w=1e-3):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()

        self.affine_layers = torch.nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(torch.nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = torch.nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

        if init_small_weights:
            for affine in self.affine_layers:
                affine.weight.data.uniform_(-init_w, init_w)
                affine.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value


class BAIL(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(BAIL, self).__init__()
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
        state = torch.tensor(state,  device=next(self.parameters()).device).float()
        action = self.forward(state)
        return action.detach().cpu().numpy()
