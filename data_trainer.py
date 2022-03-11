from einops import rearrange
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from dataset import Kinetics
import data_transform as T

class DataAugmentation(nn.Module):
	"""Module to perform data augmentation using Kornia on torch tensors."""

	def __init__(self, norm_transform, aug_transform=None, to_tensor=None):
		super().__init__()
		self.normalize = norm_transform
		self.aug_transform = aug_transform
		self.to_tensor = to_tensor

	@torch.no_grad()
	def forward(self, x, b):
		if self.to_tensor is not None:
			#print('before to tensor:', x.dtype)
			x = self.to_tensor(x)
			#print('after to tensor:', x.dtype)
		if self.aug_transform is not None:
			x = rearrange(x, '(b t) c h w -> b t c h w', b=b)
			x = self.aug_transform(x)
			x = rearrange(x, 'b t c h w -> (b t) c h w')
		#print('before norm:', x.dtype, x.min(), x.max())
		x = self.normalize(x)
		#print('after norm:', x.dtype, x.min(), x.max())
		
		return x


class Collator(object):

	def __init__(self, objective):
		self.objective = objective
	
	def collate(self, minibatch):
		image_list = []
		label_list = []
		mask_list = []
		marker_list = []
		for record in minibatch:
			# Filter out records that aren't load correctly.
			if record[-1] == -1:
				del record
				continue
			else:
				image_list.append(record[0])
				label_list.append(record[1])
				if self.objective == 'mim':
					mask_list.append(record[2])
					marker_list.append(record[3])
		minibatch = []
		minibatch.append(torch.stack(image_list))
		# need to ident whether is a  list of interge
		if self.objective == 'mim':
			minibatch.append(torch.stack(label_list))
			minibatch.append(torch.stack(mask_list))
			minibatch.append(marker_list)
		else:
			label = np.stack(label_list)
			minibatch.append(torch.from_numpy(label))
		
		return minibatch


class KineticsDataModule(pl.LightningDataModule):
	def __init__(self, 
				 configs,
				 train_ann_path,
				 val_ann_path=None,
				 test_ann_path=None,
				 ):
		super().__init__()
		self.train_ann_path = train_ann_path
		self.val_ann_path = val_ann_path
		self.test_ann_path = test_ann_path
		self.configs = configs

	def get_dataset(self, annotation_path, transform, temporal_sample):
		dataset = Kinetics(
			self.configs,
			annotation_path,
			transform=transform,
			temporal_sample=temporal_sample)
		
		return dataset

	def setup(self, stage):
		if self.configs.objective == 'mim':
			scale = (0.5, 1.0)
			color_jitter = None
		else:
			color_jitter = 0.4
			scale = None
		
		if self.configs.data_statics == 'imagenet':
			mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
		elif self.configs.data_statics == 'kinetics':
			mean, std = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)
		else:
			mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
		
		train_transform = T.create_video_transform(
			input_size=self.configs.img_size,
			is_training=True,
			scale=scale,
			hflip=0.5,
			color_jitter=color_jitter,
			auto_augment=self.configs.auto_augment,
			interpolation='bicubic',
			mean=mean,
			std=std)
		train_temporal_sample = T.TemporalRandomCrop(
			self.configs.num_frames * self.configs.frame_interval)
			
		self.train_dataset = self.get_dataset(
			self.train_ann_path,
			train_transform,
			train_temporal_sample)
		
		if self.val_ann_path is not None:
			val_transform = T.create_video_transform(
				input_size=self.configs.img_size,
				is_training=False,
				interpolation='bicubic',
				mean=mean,
				std=std)
			val_temporal_sample = T.TemporalRandomCrop(
				self.configs.num_frames * self.configs.frame_interval)
			self.val_dataset = self.get_dataset(
				self.val_ann_path,
				val_transform,
				val_temporal_sample)
		
		if self.test_ann_path is not None:
			# need to update
			test_transform = T.Compose([
				T.Resize(scale_range=(-1, 256)),
				T.ThreeCrop(size=self.configs.img_size),
				T.ToTensor(),
				T.Normalize(mean, std),
				])
			test_temporal_sample = T.TemporalRandomCrop(
				self.configs.num_frames * self.configs.frame_interval)
			self.test_dataset = self.get_dataset(
				self.test_ann_path,
				test_transform,
				test_temporal_sample)

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.configs.batch_size,
			num_workers=self.configs.num_workers,
			#collate_fn=Collator(self.objective).collate,
			shuffle=True,
			drop_last=True, 
			pin_memory=True
		)
	
	def val_dataloader(self):
		if self.val_ann_path is not None:
			return DataLoader(
				self.val_dataset,
				batch_size=self.configs.batch_size,
				num_workers=self.configs.num_workers,
				#collate_fn=Collator(self.objective).collate,
				shuffle=False,
				drop_last=False,
			)
	
	def test_dataloader(self):
		if self.test_ann_path is not None:
			return DataLoader(
				self.test_dataset,
				batch_size=self.configs.batch_size,
				num_workers=self.configs.num_workers,
				#collate_fn=Collator(self.objective).collate,
				shuffle=False,
				drop_last=False,
			)
	
	'''
	def on_after_batch_transfer(self, batch, dataloader_idx):
		if not self.trainer.testing:
			b, t, *_, = *batch[0].shape,
			video = rearrange(batch[0], 'b t c h w -> (b t) c h w')
		else:
			# n - the num crops of images
			b, n, t, *_, = *batch[0].shape,
			video = rearrange(batch[0], 'b n t c h w -> (b n t) c h w')
		
		if self.trainer.training and self.train_aug_transform is not None:
			video = self.train_aug_transform(video, b)
		elif self.trainer.validating and self.val_aug_transform is not None:
			video = self.val_aug_transform(video, b)
		elif self.trainer.testing and self.test_aug_transform is not None:
			video = self.test_aug_transform(video, b)
		
		batch[0] = rearrange(video, '(b t) c h w -> b t c h w', t=t)
		return batch
	'''