import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(parentdir))

from dataloader import TrajectoryDataset
from args import get_args
import numpy as np

args = get_args()
dataset = TrajectoryDataset(args.dataset_path, args.file_name, Threshold=0)

len = [len(traj) for traj in dataset.Trajectories]
len = np.array(len)

print(len.mean(), len.max(), len.min())

from matplotlib import pyplot as plt

plt.figure(figsize = (3, 5))
plt.boxplot(len, patch_artist=True, showmeans=True, labels = [''], medianprops={'linestyle':'--', 'color':'red'}, boxprops={'color':'black', 'facecolor': 'steelblue'})
plt.title("Distribution of Trajectory Lengths")
plt.ylabel("Length")
plt.savefig("Trajectory_Lengths.png")
plt.show()