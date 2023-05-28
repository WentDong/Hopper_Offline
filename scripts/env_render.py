import gym
from gym.envs.mujoco.hopper import HopperEnv
from dataloader import TrajectoryDataset
from args import get_args
import numpy as np
class MyHopperEnv(HopperEnv):
	def reset_model(self, state = None):
		if state is None:
			qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq
			)
			qvel = self.init_qvel + self.np_random.uniform(
				low=-0.005, high=0.005, size=self.model.nv
			)
			self.set_state(qpos, qvel)
			return self._get_obs()
		else:
			qpos = np.array([0.] + state[:5].tolist())
			qvel = state[5:]
			self.set_state(qpos, qvel)
			return self._get_obs()

if __name__ =="__main__":
	env = MyHopperEnv()
	state = env.reset()
	print(state)
	print(env.init_qpos, env.init_qvel)
	print(env.model.nq, env.model.nv)
	print(env.sim.get_state())

	args = get_args()
	dataset = TrajectoryDataset(args.dataset_path, args.file_name)
	Traj = dataset.__getitem__(0)
	state = Traj[0][0]
	print(state,state.shape)
	env.reset_model(state)
	env.render()
	for i in range(len(Traj)):
		action = Traj[i][2]

		next_state, reward, done, _ = env.step(action)
		print(np.sum(next_state-Traj[i][1]))
		print(reward-Traj[i][3])
		env.render()

