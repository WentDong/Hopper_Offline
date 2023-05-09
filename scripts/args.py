import argparse

def get_args():
	"""Get command line arguments"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file_name', default="Robust_Hopper-v4_0_10w", help='Dataset File to read')
	parser.add_argument('-d', '--dataset_path', default = "data/Hopper-v4", help='Directory of dataset')
	parser.add_argument('-b', '--batch_size', default=64, help='Batch size for training')
	parser.add_argument('-n', '--n_epochs', default=30, help='Number of epochs to train for')
	parser.add_argument('--lr', default=1e-3, help='Learning rate for training')
	parser.add_argument("--device", default="cpu", help="Device to run on")
	parser.add_argument("--save_dir", default="models", help="Directory to save models")
	args = parser.parse_args()
	return args