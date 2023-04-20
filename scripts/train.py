import torch
import numpy as np
from scripts.dataloader import D4RLTrajectoryDataset

if __name__ == "__main__":
    args = get_args()
    dataset = D4RLTrajectoryDataset(args.dataset_path, args.file_name)
    