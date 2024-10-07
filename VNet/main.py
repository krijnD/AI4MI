import argparse
from pathlib import Path
from pprint import pprint
import numpy as np
import torch
from operator import itemgetter

from torch.utils.data import DataLoader

import config
from torchvision import transforms

from ENet.dataset import SliceDataset
from utils import class2one_hot



def setup(args):

	# Set up device
	gpu = args.gpu and torch.cuda.is_available()
	device = torch.device('cuda' if gpu else 'cpu')
	print(f">>> Using device: {device}")

	n_classes = config.params.datasets_params[args.dataset]['K']
	batch_size = config.params.datasets_params[args.dataset]['B']

	# Set up the network
	pass


	# Set up the optimizer
	pass


	# Set up the data loaders
	img_transform = transforms.Compose([
		lambda img: img.convert('L'), # To grayscale
		lambda img: np.array(img)[np.newaxis, ...], # Convert the PIL Image to a NumPy Array and Add a Channel Dimension
		lambda nd: nd / 255,  # max <= 1 (normalize)
		lambda nd: np.round((255 / nd.max()) * nd),  # TODO: increase brightness of dark pictures in preprocess!!!
		lambda nd: torch.tensor(nd, dtype=torch.float32) # Convert the NumPy Array to a PyTorch Tensor
	])

	gt_transform = transforms.Compose([
		lambda img: np.array(img)[...], # Convert the PIL Image to a NumPy Array
		lambda nd: nd / (255 / (n_classes - 1)) if n_classes != 5 else nd / 63,  # max <= K-1
		lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
		lambda t: class2one_hot(t, K=n_classes), # Go from (B, H, W) to (B, K, H, W) in one hot encoding
		itemgetter(0)
	])

	root_dir = Path("data") / args.dataset

	# Set up the data loaders
	train_set = SliceDataset('train',
							 root_dir,
							 img_transform=img_transform,
							 gt_transform=gt_transform,
							 debug=args.debug)
	train_loader = DataLoader(train_set,
							  batch_size=B,
							  num_workers=args.num_workers,
							  shuffle=True)

	val_set = SliceDataset('val',
						   root_dir,
						   img_transform=img_transform,
						   gt_transform=gt_transform,
						   debug=args.debug)
	val_loader = DataLoader(val_set,
							batch_size=B,
							num_workers=args.num_workers,
							shuffle=False)

	args.dest.mkdir(parents=True, exist_ok=True)

	return None, None, None, None, None, None


def runTraining(args):
	print(f">>> Setting up to train on {args.dataset} with {args.mode}")

	net, optimizer, device, train_loader, val_loader, K = setup(args)

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--epochs', default=200, type=int)
	parser.add_argument('--dataset', default='TOY2', choices=["SEGTHOR"])
	parser.add_argument('--mode', default='full', choices=['partial', 'full'])
	parser.add_argument('--dest', type=Path, required=True,
						help="Destination directory to save the results (predictions and weights).")

	parser.add_argument('--num_workers', type=int, default=0)
	parser.add_argument('--gpu', action='store_true')
	parser.add_argument('--debug', action='store_true',
						help="Keep only a fraction (10 samples) of the datasets, "
							 "to test the logic around epochs and logging easily.")
	# Add the new --loss argument
	parser.add_argument('--loss', default='CrossEntropy', choices=['CrossEntropy', 'DiceLoss', 'CombinedLoss'],
						help="Loss function to use during training.")
	# Add optional arguments for loss weights
	parser.add_argument('--weight_ce', type=float, default=1.0,
						help="Weight for Cross Entropy loss in CombinedLoss.")
	parser.add_argument('--weight_dice', type=float, default=1.0,
						help="Weight for Dice loss in CombinedLoss.")

	args = parser.parse_args()

	pprint(args)

	runTraining(args)



if __name__ == "__main__":
	main()