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
from timm.loss import SoftTargetCrossEntropy

import utils
from mixup import Mixup
from optimizer import build_optimizer
from transformer import ClassificationHead
from video_transformer import TimeSformer, ViViT, MaskFeat

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
				 configs,
				 trainer,
				 ckpt_dir,
				 do_eval,
				 do_test,
				 n_crops=3):
		super().__init__()
		self.configs = configs
		self.trainer = trainer

		# build models
		if self.configs.objective =='mim': 
			self.model = MaskFeat(pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2]], feature_dim=2*2*2*3*9)
		else: # supervised
			# load pretrain weights from pretrained weight path and model.init_weights method
			if self.configs.arch == 'vivit':
				self.model = ViViT(
					pretrain_pth=self.configs.pretrain_pth,
					weights_from=self.configs.weights_from,
					img_size=self.configs.img_size,
					num_frames=self.configs.num_frames,
					attention_type=self.configs.attention_type)
			elif self.configs.arch == 'timesformer':
				self.model = TimeSformer(
					pretrain_pth=self.configs.pretrain_pth,
					weights_from=self.configs.weights_from,
					img_size=self.configs.img_size,
					num_frames=self.configs.num_frames,
					attention_type=self.configs.attention_type)
			else: # arch-mvit
				self.model = MaskFeat(
					pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2]], 
					feature_dim=2*2*2*3*9,
					pretrain_pth=self.configs.pretrain_pth,
					img_size=self.configs.img_size,
					num_frames=self.configs.num_frames)
				for name, param in self.model.decoder_pred.named_parameters():
					param.requires_grad = False
		
			self.cls_head = ClassificationHead(
				self.configs.num_class, self.model.embed_dims, eval_metrics=self.configs.eval_metrics)
			
			self.max_top1_acc = 0
			self.train_top1_acc = Accuracy()
			self.train_top5_acc = Accuracy(top_k=5)
			if self.configs.mixup:
				self.mixup_fn = Mixup(num_classes=self.configs.num_class)
				self.loss_fn = SoftTargetCrossEntropy()
			else:
				self.loss_fn = nn.CrossEntropyLoss()

		# common
		self.iteration = 0
		self.data_start = 0
		self.ckpt_dir = ckpt_dir
		self.do_eval = do_eval
		self.do_test = do_test
		if self.do_eval:
			self.val_top1_acc = Accuracy()
			self.val_top5_acc = Accuracy(top_k=5)
		if self.do_test:
			self.n_crops = n_crops
			self.test_top1_acc = Accuracy()
			self.test_top5_acc = Accuracy(top_k=5)
	
	@torch.jit.ignore
	def no_weight_decay_keywords(self):
		return {'pos_embed', 'cls_token', 'mask_token'}

	def configure_optimizers(self):
		# build optimzer
		is_pretrain = not (self.configs.objective == 'supervised')
		if self.configs.objective == 'supervised' and self.configs.eval_metrics == 'linear_prob':
			model = self.cls_head.module if hasattr(self.cls_head, 'module') else self.cls_head
			optimizer = build_optimizer(self.configs, model, is_pretrain=is_pretrain)
		else:
			optimizer = build_optimizer(self.configs, self, is_pretrain=is_pretrain)
		
		# lr schedule
		lr_scheduler = None
		lr_schedule = self.configs.lr_schedule 
		if lr_schedule == 'multistep':
			lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
														  milestones=[5, 11],
														  gamma=0.1)
		elif lr_schedule == 'cosine':
			lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
														  num_warmup_steps=self.configs.warmup_epochs, 
														  num_training_steps=self.trainer.max_epochs,
														  base_lr=self.configs.lr,
														  min_lr=self.configs.min_lr,
														  objective=self.configs.objective)
		return [optimizer], [lr_scheduler]

	def parse_batch(self, batch, train):
		if self.configs.objective == 'mim':
			inputs, labels, mask, cube_marker, =  *batch,
			return inputs, labels, mask, cube_marker
		else:
			inputs, labels, = *batch,
			if self.configs.mixup and train:
				inputs, labels = self.mixup_fn(inputs, labels)
			return inputs, labels

	# epoch schedule
	def _get_momentum(self, base_value, final_value):
		return final_value - (final_value - base_value) * (math.cos(math.pi * self.trainer.current_epoch / self.trainer.max_epochs) + 1) / 2

	def _weight_decay_update(self):
		for i, param_group in enumerate(self.optimizers().optimizer.param_groups):
			if i == 1:  # only the first group is regularized
				param_group["weight_decay"] = self._get_momentum(base_value=self.configs.weight_decay, final_value=self.configs.weight_decay_end)

	def clip_gradients(self, clip_grad, norm_type=2):
		layer_norm = []
		if self.configs.objective == 'supervised' and self.configs.eval_metrics == 'linear_prob':
			model_wo_ddp = self.cls_head.module if hasattr(self.cls_head, 'module') else self.cls_head
		else:
			model_wo_ddp = self.module if hasattr(self, 'module') else self
		for name, p in model_wo_ddp.named_parameters():
			if p.grad is not None:
				param_norm = torch.norm(p.grad.detach(), norm_type)
				layer_norm.append(param_norm)
				if clip_grad:
					clip_coef = clip_grad / (param_norm + 1e-6)
					if clip_coef < 1:
						p.grad.data.mul_(clip_coef)
		total_grad_norm = torch.norm(torch.stack(layer_norm), norm_type)
		return total_grad_norm
	
	def log_step_state(self, data_time, top1_acc=0, top5_acc=0):
		self.log("time",float(f'{time.perf_counter()-self.data_start:.3f}'),prog_bar=True)
		self.log("data_time", data_time, prog_bar=True)
		if self.configs.objective == 'supervised':
			self.log("top1_acc",top1_acc,on_step=True,on_epoch=False,prog_bar=True)
			self.log("top5_acc",top5_acc,on_step=True,on_epoch=False,prog_bar=True)

		return None

	def get_progress_bar_dict(self):
		# don't show the version number
		items = super().get_progress_bar_dict()
		items.pop("v_num", None)
		
		return items

	# Trainer Pipeline
	def training_step(self, batch, batch_idx):
		data_time = float(f'{time.perf_counter() - self.data_start:.3f}')
		if self.configs.objective == 'mim':
			inputs, labels, mask, cube_marker = self.parse_batch(batch, train=True)
			preds, loss = self.model(inputs, labels, mask, cube_marker)
			self.log_step_state(data_time)
			return {'loss': loss, 'data_time': data_time}
		else:
			inputs, labels = self.parse_batch(batch, train=True)
			if self.configs.eval_metrics == 'linear_prob':
				with torch.no_grad():
					self.model.eval()
					preds = self.model(inputs)
			else:
				if self.configs.arch == 'mvit':
					preds = self.model.forward_features(inputs)[:, 0]
				else:
					preds = self.model(inputs)
			preds = self.cls_head(preds)
			loss = self.loss_fn(preds, labels)
			if self.configs.mixup:
				top1_acc = self.train_top1_acc(preds.softmax(dim=-1), labels.argmax(-1))
				top5_acc = self.train_top5_acc(preds.softmax(dim=-1), labels.argmax(-1))
			else:
				top1_acc = self.train_top1_acc(preds.softmax(dim=-1), labels)
				top5_acc = self.train_top5_acc(preds.softmax(dim=-1), labels)
			self.log_step_state(data_time, top1_acc, top5_acc)
			return {'loss': loss, 'data_time': data_time}
	
	def on_after_backward(self):
		param_norms = self.clip_gradients(self.configs.clip_grad)
		self._weight_decay_update()
		# log learning daynamic
		lr = self.optimizers().optimizer.param_groups[0]['lr']
		self.log("lr",lr,on_step=True,on_epoch=False,prog_bar=True)
		self.log("grad_norm",param_norms,on_step=True,on_epoch=False,prog_bar=True)
	
	def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
		optimizer_closure, on_tpu, using_native_amp, using_lbfgs):

		optimizer.step(closure=optimizer_closure)
		self.data_start = time.perf_counter()
		self.iteration += 1

	def training_epoch_end(self, outputs):
		timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
		if self.configs.objective == 'supervised':
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

		if self.configs.objective != 'supervised' and (self.trainer.current_epoch+1) % self.configs.save_ckpt_freq == 0:
			save_path = osp.join(self.ckpt_dir,
								 f'{timestamp}_'+
								 f'ep_{self.trainer.current_epoch}.pth')
			self.trainer.save_checkpoint(save_path)

	def validation_step(self, batch, batch_indx):
		if self.do_eval:
			inputs, labels = self.parse_batch(batch, train=False)
			if self.configs.eval_metrics == 'linear_prob':
				with torch.no_grad():
					preds = self.model(inputs)
			else:
				if self.configs.arch == 'mvit':
					preds = self.model.forward_features(inputs)[:, 0]
				else:
					preds = self.model(inputs)
			preds = self.cls_head(preds)
			
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
			preds = preds.view(-1, self.n_crops, self.configs.num_class).mean(1)

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
