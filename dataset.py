import glob
import json
import os
import os.path as osp

import decord
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from einops import rearrange
from skimage.feature import hog
from mask_generator import CubeMaskGenerator

class_labels_map = None
cls_sample_cnt = None

def numpy2tensor(x):
	return torch.from_numpy(x)

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

	out = (data.view(-1, shape[-1]) * std) + mean

	return out.view(shape)


def show_processed_image(imgs, save_dir, index=0):
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
			#img = denormalize(img, mean, std).cpu().numpy()
			#img = (img * 255).astype(np.uint8)
			img = img.cpu().numpy().astype(np.uint8)
			ax.imshow(np.asarray(img))
			ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

	plt.tight_layout()
	filename = osp.join(save_dir, f'clip_transformed_b{index}.png')
	plt.savefig(filename)
 

def extract_hog_features(image):
	hog_features_r = hog(image[:,:,0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2', feature_vector=False)
	hog_features_g = hog(image[:,:,1], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2', feature_vector=False)
	hog_features_b = hog(image[:,:,2], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2', feature_vector=False) #visualize=True
	hog_features = np.concatenate([hog_features_r,hog_features_g,hog_features_b], axis=-1)
	hog_features = rearrange(hog_features, '(ph dh) (pw dw) ch cw c -> ph pw (dh dw ch cw c)', ph=14, pw=14)
	return hog_features


def load_annotation_data(data_file_path):
	with open(data_file_path, 'r') as data_file:
		return json.load(data_file)


def get_class_labels(num_class, anno_pth='./k400_classmap.json'):
	global class_labels_map, cls_sample_cnt
	
	if class_labels_map is not None:
		return class_labels_map, cls_sample_cnt
	else:
		cls_sample_cnt = {}
		class_labels_map = load_annotation_data(anno_pth)
		for cls in class_labels_map:
			cls_sample_cnt[cls] = 0
		return class_labels_map, cls_sample_cnt


def load_annotations(ann_file, num_class, num_samples_per_cls):
	dataset = []
	class_to_idx, cls_sample_cnt = get_class_labels(num_class)
	with open(ann_file, 'r') as fin:
		for line in fin:
			line_split = line.strip().split('\t')
			sample = {}
			idx = 0
			# idx for frame_dir
			frame_dir = line_split[idx]
			sample['video'] = frame_dir
			idx += 1
								
			# idx for label[s]
			label = [x for x in line_split[idx:]]
			assert label, f'missing label in line: {line}'
			assert len(label) == 1
			class_name = label[0]
			class_index = int(class_to_idx[class_name])
			
			# choose a class subset of whole dataset
			if class_index < num_class:
				sample['label'] = class_index
				if cls_sample_cnt[class_name] < num_samples_per_cls:
					dataset.append(sample)
					cls_sample_cnt[class_name]+=1

	return dataset


class DecordInit(object):
	"""Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

	def __init__(self, num_threads=1, **kwargs):
		self.num_threads = num_threads
		self.ctx = decord.cpu(0)
		self.kwargs = kwargs
		
	def __call__(self, filename):
		"""Perform the Decord initialization.
		Args:
			results (dict): The resulting dict to be modified and passed
				to the next transform in pipeline.
		"""
		reader = decord.VideoReader(filename,
									ctx=self.ctx,
									num_threads=self.num_threads)
		return reader

	def __repr__(self):
		repr_str = (f'{self.__class__.__name__}('
					f'sr={self.sr},'
					f'num_threads={self.num_threads})')
		return repr_str
		

class Kinetics(torch.utils.data.Dataset):
	"""Load the Kinetics video files
	
	Args:
		annotation_path (string): Annotation file path.
		num_class (int): The number of the class.
		num_samples_per_cls (int): the max samples used in each class.
		target_video_len (int): the number of video frames will be load.
		align_transform (callable): Align different videos in a specified size.
		temporal_sample (callable): Sample the target length of a video.
	"""

	def __init__(self,
				 annotation_path,
				 num_class, 
				 num_samples_per_cls,
				 objective='mim',
				 target_video_len=32,
				 align_transform=None,
				 aug_transform=None,
				 temporal_sample=None):
		self.data = load_annotations(annotation_path, num_class, num_samples_per_cls)

		self.align_transform = align_transform
		self.aug_transform = aug_transform
		self.temporal_sample = temporal_sample
		self.target_video_len = target_video_len
		self.objective = objective
		self.v_decoder = DecordInit()

		#mask
		if objective == 'mim':
			self.mask_generator = CubeMaskGenerator(input_size=(8,14,14),min_num_patches=16)

	def __getitem__(self, index):
		path = self.data[index]['video']
		try:
			v_reader = self.v_decoder(path)
		except:
			return None, None, None, -1
		
		# Sampling video frames
		total_frames = len(v_reader)
		start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
		if end_frame_ind-start_frame_ind < self.target_video_len:
			return None, None, None, -1
		frame_indice = np.linspace(0, end_frame_ind-start_frame_ind-1, 
								   self.target_video_len, dtype=int)
		try:
			video = v_reader.get_batch(frame_indice).asnumpy()
			del v_reader
		except:
			del v_reader
			return None, None, None, -1
		
		# Video align transform: T C H W
		with torch.no_grad():
			video = torch.from_numpy(video).permute(0,3,1,2)
			if self.align_transform is not None:
				self.align_transform.randomize_parameters()
				video = self.align_transform(video)

		# Label (depends)
		if self.objective == 'mim':
			label = np.stack(list(map(extract_hog_features, video.permute(0,2,3,1).numpy())), axis=0) # T H W C -> T H' W' C'
			mask, cube_marker = self.mask_generator() # T' H' W'
		else:
			label = self.data[index]['label']
		
		#if self.aug_transform is not None:
		#	video = self.aug_transform(video)
		if self.objective == 'mim':
			return video, numpy2tensor(label), numpy2tensor(mask), cube_marker
		else:
			return video, label

	def __len__(self):
		return len(self.data)


if __name__ == '__main__':
	# Unit test for loading video and computing time cost
	import data_transform as T
	import time
	path = './test.mp4'
	mask_generator = CubeMaskGenerator(input_size=(8,14,14),min_num_patches=16)

	counts = 1
	while True:
		if counts > 100:
			break
		start_time = time.perf_counter()
		v_decoder = DecordInit()
		v_reader = v_decoder(path)
		# Sampling video frames
		total_frames = len(v_reader)
		align_transform = T.Compose([
			T.RandomResizedCrop(size=(224, 224), area_range=(0.5, 1.0), interpolation=3), #InterpolationMode.BICUBIC
			T.Flip(),
			])
		temporal_sample = T.TemporalRandomCrop(16*4)
		start_frame_ind, end_frame_ind = temporal_sample(total_frames)
		frame_indice = np.linspace(0, end_frame_ind-start_frame_ind-1, 
								   16, dtype=int)
		video = v_reader.get_batch(frame_indice).asnumpy()
		del v_reader
		
		# Video align transform: T C H W
		with torch.no_grad():
			video = torch.from_numpy(video).permute(0,3,1,2)
			align_transform.randomize_parameters()
			video = align_transform(video)
		#label = np.stack(list(map(extract_hog_features, video.permute(0,2,3,1).numpy())), axis=0) # T H W C -> T H' W' C'
		_, hog_image = hog(video.permute(0,2,3,1).numpy()[0][:,:,2], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2', feature_vector=False, visualize=True)
		mask, cube_marker = mask_generator() # T' H' W'
		counts += 1
		print(f'{(time.perf_counter()-start_time):.3f}')
	print('finish')
	#_, hog_image = hog(video.permute(0,2,3,1).numpy()[0][:,:,2], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2', feature_vector=False, visualize=True)
	#from skimage import io
	#io.imsave('./test_img_hog.jpg',hog_image)
	#show_processed_image(video.permute(0,2,3,1), save_dir='./')
