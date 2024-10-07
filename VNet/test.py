import torch
from PIL import Image
import numpy as np
from operator import itemgetter
from ENet.utils import class2one_hot


def main():

	path_gt = "../data/SEGTHOR/train/gt/Patient_04_0029.png"
	path_img = "../data/SEGTHOR/train/img/Patient_04_0029.png"
	gt = Image.open(path_gt)
	img = Image.open(path_img)


	# plot the two images with overlap
	gt = np.array(gt)[...]



	np_gt = np.array(gt)[...]
	print(np_gt.max(), np_gt.min(), np.unique(np_gt))
	K = 5
	np_gt = np_gt / (255 / (K - 1)) if K != 5 else np_gt / 63

	print(np_gt.max(), np_gt.min(), np.unique(np_gt))

	np_gt = torch.tensor(np_gt, dtype=torch.int64)[None, ...]

	np_gt = class2one_hot(np_gt, K=K)

	print(np_gt.shape)

	# seperate the 5 images
	# np_gt = np_gt[0]
	#
	# for i in range(5):
	# 	# tensor to array
	# 	test = np_gt[i].numpy()
	# 	test = test * 255
	# 	img = Image.fromarray(test)
	# 	img.show()





if __name__ == "__main__":
	main()