import time
import os
import os.path as osp

import numpy as np 
import torch
import matplotlib.pyplot as plt
import torch.distributed as dist
from pytorch_lightning.utilities.distributed import rank_zero_only

@rank_zero_only
def print_on_rank_zero(content):
	if is_main_process():	
		print(content)
	
def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True

def get_world_size():
	if not is_dist_avail_and_initialized():
		return 1
	return dist.get_world_size()

def get_rank():
	if not is_dist_avail_and_initialized():
		return 0
	return dist.get_rank()

def is_main_process():
	return get_rank() == 0

def timeit_wrapper(func, *args, **kwargs):
	start = time.perf_counter()
	func_return_val = func(*args, **kwargs)
	end = time.perf_counter()
	return func_return_val, float(f'{end - start:.4f}')

def show_trainable_params(named_parameters):
	for name, param in named_parameters:
		print(name, param.size())

def build_param_groups(model):
	params_no_decay = []
	params_has_decay = []
	params_no_decay_name = []
	params_decay_name = []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if len(param) == 1 or name.endswith('.bias'): 
			params_no_decay.append(param)
			params_no_decay_name.append(name)
		else:
			params_has_decay.append(param)
			params_decay_name.append(name)

	param_groups = [
					{'params': params_no_decay, 'weight_decay': 0},
					{'params': params_has_decay},
					]
	print_on_rank_zero(f'params_no_decay_name: {params_no_decay_name} \n params_decay_name: {params_decay_name}')
	return param_groups


def denormalize(data, mean, std):
	"""Denormalize an image/video tensor with mean and standard deviation.

	Args:
		input: Image tensor of size : (H W C).
		mean: Mean for each channel.
		std: Standard deviations for each channel.

	Return:
		Denormalised tensor with same size as input : (H W C).
	"""
	shape = data.shape

	if isinstance(mean, tuple):
		mean = np.array(mean, dtype=float)
		mean = torch.tensor(mean, device=data.device, dtype=data.dtype)

	if isinstance(std, tuple):
		std = np.array(std, dtype=float)
		std = torch.tensor(std, device=data.device, dtype=data.dtype)

	if mean.shape:
		mean = mean[None, :]
	if std.shape:
		std = std[None, :]

	out = (data.contiguous().view(-1, shape[-1]) * std) + mean

	return out.view(shape)


def show_processed_image(imgs, save_dir, mean, std, index=0):
	"""Plot the transformed images into figure and save to disk.
	
	Args:
		imgs: Image tensor of size : (T H W C).
		save_dir: The path to save the images.
		index: The index of current clips.
	"""
	os.makedirs(save_dir, exist_ok=True)
	if not isinstance(imgs[0], list):
		imgs = [imgs]
		
	num_show_clips = 5
	num_rows = len(imgs)
	num_cols = num_show_clips
	fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
	for row_idx, row in enumerate(imgs):
		row = row[:num_show_clips]
		for col_idx, img in enumerate(row):
			ax = axs[row_idx, col_idx]
			img = denormalize(img, mean, std).cpu().numpy()
			img = (img * 255).astype(np.uint8)
			#img = img.cpu().numpy().astype(np.uint8)
			ax.imshow(np.asarray(img))
			ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

	plt.tight_layout()
	filename = osp.join(save_dir, f'clip_transformed_b{index}.png')
	plt.savefig(filename)