import os.path as osp
import math
import time

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchmetrics import Accuracy

from transformer import ClassificationHead
from utils import timeit_wrapper, print_on_rank_zero
from video_transformer import TimeSformer, ViViT, MaskFeat

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

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, base_lr, objective, min_lr=5e-5, last_epoch=-1):
	""" Create a schedule with a learning rate that decreases following the
	values of the cosine function between 0 and `pi * cycles` after a warmup
	period during which it increases linearly between 0 and base_lr.
	"""
	# step means epochs here
	def lr_lambda(current_step):
		current_step += 1
		if current_step <= num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps)) # * base_lr 
		progress = min(float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps)), 1)
		if objective == 'mim':
			return 0.5 * (1. + math.cos(math.pi * progress))
		else:
			factor = 0.5 * (1. + math.cos(math.pi * progress))
			return factor*(1 - min_lr/base_lr) + min_lr/base_lr

	return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

class VideoTransformer(pl.LightningModule):

	def __init__(self, 
				 lr,
				 n_crops,
				 log_interval,
				 num_classes,
				 trainer,
				 ckpt_dir,
				 do_eval,
				 do_test,
				 objective,
				 arch,
				 save_ckpt_freq,
				 lr_schedule,
				 optim_type,
				 **model_kwargs):
		super().__init__()
		self.objective = objective
		self.arch = arch
		if self.objective =='mim': 
			self.model = MaskFeat(pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2]], feature_dim=2*2*2*3*9)
			#self.no_decay_layer = ["pos_embed_spatial", "pos_embed_temporal", "pos_embed_class", "cls_token", 'mask_token']
			self.save_ckpt_freq = save_ckpt_freq
		else:
			if arch == 'vivit':
				self.model = ViViT(**model_kwargs)
			else:
				self.model = TimeSformer(**model_kwargs)
			self.cls_head = ClassificationHead(num_classes, model_kwargs['embed_dims'])
			#self.no_decay_layer = ['cls_token','pos_embed','time_embed']
			self.loss_fn = nn.CrossEntropyLoss()
			self.max_top1_acc = 0
			self.train_top1_acc = Accuracy()
			self.train_top5_acc = Accuracy(top_k=5)

		self.trainer = trainer
		self.lr_schedule = lr_schedule
		self.optim_type = optim_type
		self.lr = lr
		self.n_crops = n_crops
		self.num_classes = num_classes
		self.log_interval = log_interval
		self.iteration = 0
		self.data_start = 0
		self.ckpt_dir = ckpt_dir
		self.do_eval = do_eval
		self.do_test = do_test
		if self.do_eval:
			self.val_top1_acc = Accuracy()
			self.val_top5_acc = Accuracy(top_k=5)
		if self.do_test:
			self.test_top1_acc = Accuracy()
			self.test_top5_acc = Accuracy(top_k=5)  

	def configure_optimizers(self):
		param_groups = build_param_groups(self)
		lr_schedule = self.lr_schedule 
		optim_type = self.optim_type

		# optimzer
		if optim_type == 'sgd':
			optimizer = optim.SGD(param_groups,
								  lr=self.lr,
								  momentum=0.9,
								  weight_decay=0.0001,
								  nesterov=True)
		else:
			optimizer = torch.optim.AdamW(param_groups, 
										  lr=self.lr,
										  betas=(0.9, 0.999),
										  weight_decay=0.05)
		# lr schedule
		lr_scheduler = None
		if lr_schedule == 'multistep':
			lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
														  milestones=[5, 11],
														  gamma=0.1)
		elif lr_schedule == 'cosine':
			if self.objective == 'mim':
				num_warmup_steps = 30
				num_training_steps = 300
			else:
				num_warmup_steps = 2
				num_training_steps = 30
			lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
														  num_warmup_steps=num_warmup_steps, 
														  num_training_steps=num_training_steps,
														  base_lr=self.lr,
														  objective=self.objective)
		return [optimizer], [lr_scheduler]

	def parse_batch(self, batch):
		if self.objective == 'mim':
			inputs, labels, mask, cube_marker, =  *batch,
			return inputs, labels, mask, cube_marker
		else:
			inputs, labels, = *batch,
			return inputs, labels
		
	def comp_grad_norm(self, norm_type):
		layer_param_scale = []
		layer_update_scale = []
		layer_norm = []
		for tag, value in self.model.named_parameters():
			tag = tag.replace('.', '/')
			if value.grad is not None:
				layer_norm.append(torch.norm(value.grad.detach(), norm_type))
			
		total_grad_norm = torch.norm(torch.stack(layer_norm), norm_type)
		return total_grad_norm
	
	def log_step_state(self, data_time, top1_acc=0, top5_acc=0):
		self.log("time",float(f'{time.perf_counter()-self.data_start:.3f}'),prog_bar=True)
		self.log("data_time", data_time, prog_bar=True)
		if self.objective == 'supervised':
			self.log("top1_acc",top1_acc,on_step=True,on_epoch=False,prog_bar=True)
			self.log("top5_acc",top5_acc,on_step=True,on_epoch=False,prog_bar=True)

		return None
	
	def get_progress_bar_dict(self):
		# don't show the version number
		items = super().get_progress_bar_dict()
		items.pop("v_num", None)
		
		return items

	def training_step(self, batch, batch_idx):
		data_time = float(f'{time.perf_counter() - self.data_start:.3f}')
		if self.objective == 'mim':
			inputs, labels, mask, cube_marker = self.parse_batch(batch)
			preds, loss = self.model(inputs, labels, mask, cube_marker)
			return {'loss': loss, 'data_time':data_time}
		else:
			inputs, labels = self.parse_batch(batch)
			preds = self.model(inputs)
			preds = self.cls_head(preds)
			
			return {'preds':preds,'labels':labels,'data_time':data_time}
	
	def on_before_optimizer_step(self, optimizer, opt_idx):
		if self.iteration % self.log_interval == self.log_interval-1:
			lr = optimizer.param_groups[0]['lr']
			grad_norm = self.comp_grad_norm(norm_type=2.0)
			self.log("lr",lr,on_step=True,on_epoch=False,prog_bar=True)
			self.log("grad_norm",grad_norm,on_step=True,on_epoch=False,prog_bar=True)

	def training_step_end(self, outputs):
		if self.objective == 'mim':
			loss, data_time = outputs['loss'], outputs['data_time']
			self.log_step_state(data_time)
		else:
			preds, labels, data_time = outputs['preds'], outputs['labels'], outputs['data_time']
			loss = self.loss_fn(preds, labels)
			top1_acc = self.train_top1_acc(preds.softmax(dim=-1), labels)
			top5_acc = self.train_top5_acc(preds.softmax(dim=-1), labels)
			self.log_step_state(data_time, top1_acc, top5_acc)
		self.iteration += 1
		self.data_start = time.perf_counter()
		return loss

	def training_epoch_end(self, outputs):
		timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
		if self.objective == 'supervised':
			mean_top1_acc = self.train_top1_acc.compute()
			mean_top5_acc = self.train_top5_acc.compute()
			self.print(f'{timestamp} - Evaluating mean ',
					   f'top1_acc:{mean_top1_acc:.3f},',
					   f'top5_acc:{mean_top5_acc:.3f} of current training epoch')
			self.train_top1_acc.reset()
			self.train_top5_acc.reset()

		# save last checkpoint
		save_path = osp.join(self.ckpt_dir, 'last_checkpoint.pth')
		self.trainer.save_checkpoint(save_path)

		if self.objective == 'mim' and (self.trainer.current_epoch+1) % self.save_ckpt_freq == 0:
			save_path = osp.join(self.ckpt_dir,
								 f'{timestamp}_'+
								 f'ep_{self.trainer.current_epoch}.pth')
			self.trainer.save_checkpoint(save_path)


	def validation_step(self, batch, batch_indx):
		if self.do_eval:
			inputs, labels = self.parse_batch(batch)
			preds = self.cls_head(self.model(inputs))
			
			self.val_top1_acc(preds.softmax(dim=-1), labels)
			self.val_top5_acc(preds.softmax(dim=-1), labels)
			self.data_start = time.perf_counter()
	
	def validation_epoch_end(self, outputs):
		if self.do_eval:
			mean_top1_acc = self.val_top1_acc.compute()
			mean_top5_acc = self.val_top5_acc.compute()
			timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
			self.print(f'{timestamp} - Evaluating mean ',
					   f'top1_acc:{mean_top1_acc:.3f}, ',
					   f'top5_acc:{mean_top5_acc:.3f} of current validation epoch')
			self.val_top1_acc.reset()
			self.val_top5_acc.reset()

			# save best checkpoint
			if mean_top1_acc > self.max_top1_acc:
				save_path = osp.join(self.ckpt_dir,
									 f'{timestamp}_'+
									 f'ep_{self.trainer.current_epoch}_'+
									 f'top1_acc_{mean_top1_acc:.3f}.pth')
				self.trainer.save_checkpoint(save_path)
				self.max_top1_acc = mean_top1_acc
			
	def test_step(self, batch, batch_idx):
		if self.do_test:
			inputs, labels = self.parse_batch(batch)
			preds = self.cls_head(self.model(inputs))
			preds = preds.view(-1, self.n_crops, self.num_classes).mean(1)

			self.test_top1_acc(preds.softmax(dim=-1), labels)
			self.test_top5_acc(preds.softmax(dim=-1), labels)
			self.data_start = time.perf_counter()
	
	def test_epoch_end(self, outputs):
		if self.do_test:
			mean_top1_acc = self.test_top1_acc.compute()
			mean_top5_acc = self.test_top5_acc.compute()
			timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
			self.print(f'{timestamp} - Evaluating mean ',
					   f'top1_acc:{mean_top1_acc:.3f}, ',
					   f'top5_acc:{mean_top5_acc:.3f} of current test epoch')
			self.test_top1_acc.reset()
			self.test_top5_acc.reset()
