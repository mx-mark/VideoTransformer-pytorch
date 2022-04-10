import json
import random

import decord
import numpy as np
import torch

from einops import rearrange
from skimage.feature import hog
from mask_generator import CubeMaskGenerator

class_labels_map = None
cls_sample_cnt = None

def temporal_sampling(frames, start_idx, end_idx, num_samples):
	"""
	Given the start and end frame index, sample num_samples frames between
	the start and end with equal interval.
	Args:
		frames (tensor): a tensor of video frames, dimension is
			`num video frames` x `channel` x `height` x `width`.
		start_idx (int): the index of the start frame.
		end_idx (int): the index of the end frame.
		num_samples (int): number of frames to sample.
	Returns:
		frames (tersor): a tensor of temporal sampled video frames, dimension is
			`num clip frames` x `channel` x `height` x `width`.
	"""
	index = torch.linspace(start_idx, end_idx, num_samples)
	index = torch.clamp(index, 0, frames.shape[0] - 1).long()
	frames = torch.index_select(frames, 0, index)
	return frames


def numpy2tensor(x):
	return torch.from_numpy(x)


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
				 configs,
				 annotation_path,
				 transform=None,
				 temporal_sample=None):
		self.configs = configs
		self.data = load_annotations(annotation_path, self.configs.num_class, self.configs.num_samples_per_cls)

		self.transform = transform
		self.temporal_sample = temporal_sample
		self.target_video_len = self.configs.num_frames
		self.objective = self.configs.objective
		self.v_decoder = DecordInit()

		# mask
		if self.objective == 'mim':
			self.mask_generator = CubeMaskGenerator(input_size=(self.target_video_len//2,14,14),min_num_patches=16)

	def __getitem__(self, index):
		while True:
			try:
				path = self.data[index]['video']
				v_reader = self.v_decoder(path)
				total_frames = len(v_reader)
				
				# Sampling video frames
				start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
				assert end_frame_ind-start_frame_ind >= self.target_video_len
				frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
				video = v_reader.get_batch(frame_indice).asnumpy()
				del v_reader
				break
			except Exception as e:
				print(e)
				index = random.randint(0, len(self.data) - 1)
		
		# Video align transform: T C H W
		with torch.no_grad():
			video = torch.from_numpy(video).permute(0,3,1,2)
			if self.transform is not None:
				if self.objective == 'mim':
					pre_transform, post_transform = self.transform
					video = pre_transform(video) # align shape
				else:
					video = self.transform(video)

		# Label (depends)
		if self.objective == 'mim':
			# old version
			'''
			mask, cube_marker = self.mask_generator() # T' H' W'
			label = np.stack(list(map(extract_hog_features, video.permute(0,2,3,1).numpy())), axis=0) # T H W C -> T H' W' C'
			'''
			# new version
			mask, cube_marker = self.mask_generator() # T' H' W'
			hog_inputs = video.permute(0,2,3,1).numpy()
			hog_features = np.zeros((self.target_video_len,14,14,2*2*3*9))
			# speed up the extraction of hog features
			for marker in cube_marker: # [[start, span]]
				start_frame, span_frame = marker
				center_index = start_frame*2 + span_frame*2//2 # fix the temporal stride to 2
				hog_features[center_index] = extract_hog_features(hog_inputs[center_index])
			label = hog_features
		else:
			label = self.data[index]['label']
		
		if self.objective == 'mim':
			if self.transform is not None:
				video = post_transform(video) # to tensor & norm
			return video, numpy2tensor(label), numpy2tensor(mask), cube_marker
		else:
			return video, label

	def __len__(self):
		return len(self.data)


if __name__ == '__main__':
	# Unit test for loading video and computing time cost
	import data_transform as T
	import time
	path = './YABnJL_bDzw.mp4'
	color_jitter = 0.4
	auto_augment = 'rand-m9-mstd0.5-inc1'
	scale = None
	mean, std = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)
	transform = T.create_video_transform(
		input_size=224,
		is_training=True,
		scale=scale,
		hflip=0.5,
		color_jitter=color_jitter,
		auto_augment=auto_augment,
		interpolation='bicubic',
		mean=mean,
		std=std)
	
	v_decoder = DecordInit()
	v_reader = v_decoder(path)
	total_frames = len(v_reader)
	target_video_len = 16
	# Sampling video frames
	temporal_sample = T.TemporalRandomCrop(target_video_len*16)
	start_frame_ind, end_frame_ind = temporal_sample(total_frames)
	frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, target_video_len, dtype=int)
	video = v_reader.get_batch(frame_indice).asnumpy()
	del v_reader
	
	# Video align transform: T C H W
	with torch.no_grad():
		video = torch.from_numpy(video).permute(0,3,1,2)
		if transform is not None:
			video = transform(video)
	
	show_processed_image(video.permute(0,2,3,1), save_dir='./', mean=mean, std=std)
	'''
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
	'''
	#_, hog_image = hog(video.permute(0,2,3,1).numpy()[0][:,:,2], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2', feature_vector=False, visualize=True)
	#from skimage import io
	#io.imsave('./test_img_hog.jpg',hog_image)
	#show_processed_image(video.permute(0,2,3,1), save_dir='./')
