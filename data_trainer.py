from einops import rearrange
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from dataset import Kinetics


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, norm_transform, aug_transform=None):
        super().__init__()
        self.normalize = norm_transform
        self.aug_transform = aug_transform

    @torch.no_grad()
    def forward(self, x, b):
    	if self.aug_transform is not None:
    		x = rearrange(x, '(b t) c h w -> b t c h w', b=b)
    		x = self.aug_transform(x)
    		x = rearrange(x, 'b t c h w -> (b t) c h w')
    	x = self.normalize(x)
    	
    	return x


class Collator(object):

    def __init__(self):
        pass
  	
    def collate(self, minibatch):
    	image_list = []
    	label_list = []
    	for record in minibatch:
    		# Filter out records that aren't load correctly.
    		if record[-1] == -1:
    			del record
    			continue
    		else:
    			image_list.append(record[0])
    			label_list.append(record[1])
    	minibatch = []
    	minibatch.append(torch.stack(image_list))
    	label = np.stack(label_list)
    	minibatch.append(torch.from_numpy(label))
    	return minibatch


class KineticsDataModule(pl.LightningDataModule):
    def __init__(self, 
    			 train_ann_path,
    			 train_align_transform,
    			 train_aug_transform,
    			 train_temporal_sample,
    			 val_ann_path=None,
    			 val_aug_transform=None,
    			 val_align_transform=None,
    			 val_temporal_sample=None,
    			 test_ann_path=None,
    			 test_align_transform=None,
    			 test_aug_transform=None,
    			 test_temporal_sample=None,
    			 num_class=600, 
    			 num_samples_per_cls=10000,
    			 target_video_len=8,
    			 batch_size=8,
    			 num_workers=4):
        super().__init__()
        self.train_ann_path = train_ann_path
        self.train_align_transform = train_align_transform
        self.train_aug_transform = train_aug_transform
        self.train_temporal_sample = train_temporal_sample
        self.val_ann_path = val_ann_path
        self.val_align_transform = val_align_transform
        self.val_aug_transform = val_aug_transform 
        self.val_temporal_sample = val_temporal_sample
        self.test_ann_path = test_ann_path
        self.test_align_transform = test_align_transform
        self.test_aug_transform = test_aug_transform
        self.test_temporal_sample = test_temporal_sample
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_video_len = target_video_len
        self.num_class = num_class 
        self.num_samples_per_cls = num_samples_per_cls

    def get_dataset(self, annotation_path, align_transform, temporal_sample):
        dataset = Kinetics(
            annotation_path,
            align_transform=align_transform,
            temporal_sample=temporal_sample,
            num_class=self.num_class, 
            num_samples_per_cls=self.num_samples_per_cls,
            target_video_len=self.target_video_len)
        
        return dataset

    def setup(self, stage):
        self.train_dataset = self.get_dataset(
        	self.train_ann_path,
        	self.train_align_transform,
        	self.train_temporal_sample)
        
        if self.val_ann_path is not None:
        	self.val_dataset = self.get_dataset(
				self.val_ann_path,
				self.val_align_transform,
				self.val_temporal_sample)
        
        if self.test_ann_path is not None:
        	self.test_dataset = self.get_dataset(
				self.test_ann_path,
				self.test_align_transform,
				self.test_temporal_sample)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=Collator().collate,
            shuffle=True,
            drop_last=True, 
            pin_memory=True
        )
    
    def val_dataloader(self):
    	if self.val_ann_path is not None:
    		return DataLoader(
    			self.val_dataset,
    			batch_size=self.batch_size,
    			num_workers=self.num_workers,
    			collate_fn=Collator().collate,
    			shuffle=False,
    			drop_last=False,
    		)
    
    def test_dataloader(self):
    	if self.test_ann_path is not None:
    		return DataLoader(
    			self.test_dataset,
    			batch_size=self.batch_size,
    			num_workers=self.num_workers,
    			collate_fn=Collator().collate,
    			shuffle=False,
    			drop_last=False,
    		)

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