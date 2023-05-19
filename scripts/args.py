import argparse

def get_args(algo="bc"):
	"""Get command line arguments"""
	algo = algo.lower()

	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file_name', default="Robust_Hopper-v4_0_10w", help='Dataset File to read')
	parser.add_argument('-d', '--dataset_path', default = "data/Hopper-v4", help='Directory of dataset')
	parser.add_argument('--hidden_dim', default=512, type = int, help='Hidden dimension of network')
	parser.add_argument('-b', '--batch_size', default=64, type = int, help='Batch size for training')
	parser.add_argument('-n', '--n_epochs', default=30, type = int, help='Number of epochs to train for')
	parser.add_argument('--lr', default=1e-3, type = float, help='Learning rate for training')
	parser.add_argument('--wd', default=0, type = float, help='Weight decay for training')
	parser.add_argument("--device", default="cuda", help="Device to run on")
	parser.add_argument("--save_dir", default="models", help="Directory to save models")
	if algo == "bail":  # BAIL
		parser.add_argument('-t', '--trajectory_truncation', default=0., type=float, help='Trajectory truncation for BAIL')
		parser.add_argument('-r', '--rollout', default=1000, type = int, help='Rollout length for MC estimation')
		parser.add_argument('-g', '--gamma', default=0.99, type = float, help='Discount factor')
		parser.add_argument('-a', '--augment_mc', default="gain", help='Augmentation method for MC estimation')
		parser.add_argument('-u', '--ue_n_epochs', default=50, type = int, help='Number of epochs to train upper envelope for')
		parser.add_argument('--ue_lr', default=3e-3, type = float, help='Learning rate for training upper envelope')
		parser.add_argument('--ue_wd', default=2e-2, type = float, help='Weight decay for training upper envelope')
		parser.add_argument('-c', '--clip_ue', action='store_true', help='Clip upper envelope')
		parser.add_argument('-i', '--detect_interval', default=10000, type = int, help='Detection interval for BAIL')
		parser.add_argument('-k', '--ue_loss_k', default=1000, type = int, help='Soft constraint for upper envelope loss')
		parser.add_argument('-p', '--select_percentage', default=0.25, type=float, help='Percentage of data to select')
		parser.add_argument('--max_timesteps', default=int(2e6), type = int, help='Max time steps to run environment for')
	args = parser.parse_args()

	if algo == "bail":
		args.setting_name = "%s_r%s_g%s_t%s" % (args.file_name, args.rollout, args.gamma, args.trajectory_truncation)
		args.setting_name += '_noaug' if not (args.augment_mc) else ''
		args.setting_name += '_augNew' if args.augment_mc == 'new' else ''
		args.data_name = args.setting_name
		args.data_name += '_Gain' if args.augment_mc == 'gain' else '_Gt'
		args.ue_setting = 'Stat_' + args.setting_name + '_lossk%s' % args.ue_loss_k

	return args