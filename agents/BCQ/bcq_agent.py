import torch
from torch import nn
import copy

class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim_VAE, latent_dim, device = 'cpu'):
		super(VAE, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.latent_dim = latent_dim
		self.device = device
		self.encoder = nn.Sequential(
			nn.Linear(state_dim + action_dim, hidden_dim_VAE[0]),
			nn.ReLU(),
			nn.Linear(hidden_dim_VAE[0], hidden_dim_VAE[1]),
			nn.ReLU(),
			nn.Linear(hidden_dim_VAE[1], latent_dim * 2)
		)
		self.decoder = nn.Sequential(
			nn.Linear(state_dim + latent_dim, hidden_dim_VAE[1]),
			nn.ReLU(),
			nn.Linear(hidden_dim_VAE[1], hidden_dim_VAE[0]),
			nn.ReLU(),
			nn.Linear(hidden_dim_VAE[0], action_dim),
			nn.Tanh()
		)
	def reparameterize(self, mu, std):
		z = mu + std * torch.randn_like(mu)
		return z
	def forward(self, states, actions):
		z = self.encoder(torch.cat([states, actions], 1))
		mu, lg_std = torch.split(z, z.size(1) // 2, dim=1)
		std = torch.exp(torch.clamp(lg_std, min = -4, max = 15))
		z = self.reparameterize(mu, std)
		return self.decoder(torch.cat([states, z], 1)), mu, std

	def call_samples(self, states):
		z = torch.randn((states.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)
		# print(z.shape, states.shape)
		return self.decoder(torch.cat([states, z], 1))

class Actor_disturb(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim, phi = 0.05, device = 'cpu'):
		super(Actor_disturb, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.phi = phi
		self.device = device
		self.actor = nn.Sequential(
			nn.Linear(state_dim + action_dim, hidden_dim[0]),
			nn.ReLU(),
			nn.Linear(hidden_dim[0], hidden_dim[1]),
			nn.ReLU(),
			nn.Linear(hidden_dim[1], action_dim),
			nn.Tanh()
		)
	def forward(self, states, actions):
		return torch.clamp(self.phi * self.actor(torch.cat([states, actions], 1)) + actions, -1, 1)
	
class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim, device = 'cpu'):
		super(Critic, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.device = device
		self.Q1 = nn.Sequential(
			nn.Linear(state_dim + action_dim, hidden_dim[0]),
			nn.ReLU(),
			nn.Linear(hidden_dim[0], hidden_dim[1]),
			nn.ReLU(),
			nn.Linear(hidden_dim[1], 1)
		)
		self.Q2 = nn.Sequential(
			nn.Linear(state_dim + action_dim, hidden_dim[0]),
			nn.ReLU(),
			nn.Linear(hidden_dim[0], hidden_dim[1]),
			nn.ReLU(),
			nn.Linear(hidden_dim[1], 1)
		)
	def forward(self, states, actions):
		return self.Q1(torch.cat([states, actions], 1)), self.Q2(torch.cat([states, actions], 1))
	def call_single(self, states, actions):
		return self.Q1(torch.cat([states, actions], 1))
	
class BCQ(nn.Module):
	def __init__(self, gamma=0.99, state_dim=11, action_dim=3, latent_dim=10, hidden_dim_VAE=[750,750], hidden_dim_Q = [400,300], Phi = 0.05, lambd = 0.7, tau = 0.005, device = 'cpu', lr=1e-3, lr_critic=None):
		super(BCQ, self).__init__()
		self.gamma = gamma
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.latent_dim = latent_dim
		self.n_samples = 10
		self.device = device
		self.lambd = lambd
		self.tau = tau

		self.VAE_net = VAE(state_dim, action_dim, hidden_dim_VAE, latent_dim, device)
		self.Actor_disturb_net = Actor_disturb(state_dim, action_dim, hidden_dim_Q, Phi, device)
		self.Critic_net = Critic(state_dim, action_dim, hidden_dim_Q, device)

		self.Critic_target = copy.deepcopy(self.Critic_net)
		self.Actor_disturb_target = copy.deepcopy(self.Actor_disturb_net)

		self.VAE_optim = torch.optim.AdamW(self.VAE_net.parameters(), lr = lr)
		self.Actor_disturb_optim = torch.optim.AdamW(self.Actor_disturb_net.parameters(), lr = lr)
		self.Critic_optim = torch.optim.AdamW(self.Critic_net.parameters(), lr = lr_critic if lr_critic is not None else lr)
		self.VAE_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.VAE_optim, mode='min', factor=0.5, patience=10, verbose=True)
		self.Actor_disturb_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.Actor_disturb_optim, mode='min', factor=0.5, patience=10, verbose=True)
		self.Critic_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.Critic_optim, mode='min', factor=0.5, patience=10, verbose=True)
		
	def forward(self, states):
		batch_states = states.reshape(-1,self.state_dim).repeat_interleave(self.n_samples, 0)
		bs = batch_states.shape[0]//self.n_samples
		actions = self.VAE_net.call_samples(batch_states)
		actions = self.Actor_disturb_net(batch_states, actions)
		Q1, Q2 = self.Critic_net(batch_states, actions)
		Q = torch.min(Q1, Q2)
		idx = Q.reshape((bs, -1)).max(1)[1]
		idx = idx.reshape(-1,1,1).expand(-1, -1, self.action_dim)
		ret = torch.gather(actions.reshape(bs,-1,self.action_dim), 1, idx).squeeze()
		return ret

	def train(self, states, actions, next_states, rewards, not_dones, select=None):
		Recon, mu, std = self.VAE_net(states, actions)
		Recon_loss = nn.functional.mse_loss(actions, Recon, reduction='none').mean(-1)
		KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2)).mean(-1)
		VAE_loss = Recon_loss + KL_loss / 2
		if select is not None:
			VAE_loss = VAE_loss[select.bool()]
		VAE_loss = VAE_loss.mean()
		KL_loss = KL_loss.mean()
		Recon_loss = Recon_loss.mean()
		self.VAE_optim.zero_grad()
		VAE_loss.backward()
		self.VAE_optim.step()
		self.VAE_sched.step(VAE_loss)
		
		q1, q2 = self.Critic_net(states, actions)

		with torch.no_grad():
			Repeat_Next_States = next_states.repeat_interleave(self.n_samples, 0) # ((n x B), 11),  [B, B, ..., B]^T
			sample_actions = self.VAE_net.call_samples(Repeat_Next_States).detach()
			sample_actions = self.Actor_disturb_target(Repeat_Next_States, sample_actions)

			# print("AAA", sample_actions.shape)
			target_Q1, target_Q2 = self.Critic_target(Repeat_Next_States, sample_actions)
			# print("BBB:", target_Q1.shape)
			target_Q = self.lambd * torch.min(target_Q1, target_Q2) + (1 - self.lambd) * torch.max(target_Q1, target_Q2)
			# print("CCC", target_Q.reshape(64,-1).max(1)[0].shape)
			target_Q = target_Q.reshape(states.shape[0], -1).max(1)[0].reshape(-1,1)
			target_Q = rewards + not_dones * target_Q * self.gamma
		
		Critic_loss = nn.functional.mse_loss(q1, target_Q) + nn.functional.mse_loss(q2, target_Q)
		self.Critic_optim.zero_grad()
		Critic_loss.backward()
		self.Critic_optim.step()
		self.Critic_sched.step(Critic_loss)
		
		actor_samples = self.VAE_net.call_samples(states)
		actor_samples = self.Actor_disturb_net(states, actor_samples)
		Actor_loss = -self.Critic_net.call_single(states, actor_samples).mean()
		self.Actor_disturb_optim.zero_grad()
		Actor_loss.backward()
		self.Actor_disturb_optim.step()
		self.Actor_disturb_sched.step(Actor_loss)

		for param, target_param in zip(self.Critic_net.parameters(), self.Critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.Actor_disturb_net.parameters(), self.Actor_disturb_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return Recon_loss.item(), KL_loss.item(), Critic_loss.item(), Actor_loss.item()

	def take_action(self, states):
		cp_states = torch.tensor(states, device=next(self.parameters()).device).float()
		return self.forward(cp_states).detach().cpu().numpy()