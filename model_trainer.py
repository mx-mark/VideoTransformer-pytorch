import os.path as osp
import time

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchmetrics import Accuracy

from transformer import ClassificationHead
from utils import timeit_wrapper
from video_transformer import TimeSformer


def show_trainable_params(named_parameters):
	for name, param in named_parameters:
		print(name, param.size())


def build_param_groups(model, no_decay_layer):
	params_no_decay = []
	params_has_decay = []
	for name, param in model.named_parameters():
		if name in no_decay_layer:
			params_no_decay.append(param)
		else:
			params_has_decay.append(param)

	param_groups = [
					{'params': params_no_decay, 'weight_decay': 0},
					{'params': params_has_decay},
                    ]
	return param_groups


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
    			 **model_kwargs):
        super().__init__()
        self.model = TimeSformer(**model_kwargs)
        self.cls_head = ClassificationHead(num_classes, model_kwargs['embed_dims'])
        self.trainer = trainer
        
        self.lr = lr
        self.n_crops = n_crops
        self.num_classes = num_classes
        self.log_interval = log_interval
        self.iteration = 0
        self.data_start = 0
        self.max_top1_acc = 0
        self.loss_fn = nn.CrossEntropyLoss()
        self.ckpt_dir = ckpt_dir
        self.no_decay_layer = ['cls_token','pos_embed','time_embed']
        
        self.do_eval = do_eval
        self.do_test = do_test
        self.train_top1_acc = Accuracy()
        self.train_top5_acc = Accuracy(top_k=5)
        if self.do_eval:
        	self.val_top1_acc = Accuracy()
        	self.val_top5_acc = Accuracy(top_k=5)
        if self.do_test:
        	self.test_top1_acc = Accuracy()
        	self.test_top5_acc = Accuracy(top_k=5)  

    def configure_optimizers(self):
        #optimizer = optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.05)
        param_groups = build_param_groups(self, self.no_decay_layer)
        optimizer = optim.SGD(param_groups,
        					  lr=self.lr,
        					  momentum=0.9,
        					  weight_decay=0.0001,
        					  nesterov=True)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
        											  milestones=[5, 11],
        											  gamma=0.1)
        return [optimizer], [lr_scheduler]

    def parse_batch(self, batch):
    	inputs = {}
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
    
    def log_step_state(self, data_time, model_time):
        self.log("time",float(f'{time.perf_counter()-self.data_start:.3f}'),prog_bar=True)
        self.log("data_time", data_time, prog_bar=True)
        self.log("model_time", model_time)
        self.log("top1_acc",self.train_top1_acc,on_step=True,on_epoch=False,prog_bar=True)
        self.log("top5_acc",self.train_top5_acc,on_step=True,on_epoch=False,prog_bar=True)
        
        return None
    
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
    	
        return items

    def training_step(self, batch, batch_idx):
        inputs, labels = self.parse_batch(batch)
        data_time = float(f'{time.perf_counter() - self.data_start:.3f}')
        preds, model_time = timeit_wrapper(self.model, inputs)
        preds = self.cls_head(preds)
        
        return {'preds':preds,'labels':labels,
        		'data_time':data_time,'model_time':model_time}
    
    def on_before_optimizer_step(self, optimizer, opt_idx):
    
        if self.iteration % self.log_interval == self.log_interval-1:
        	lr = optimizer.param_groups[0]['lr']
        	grad_norm = self.comp_grad_norm(norm_type=2.0)
        	self.log("lr",lr,on_step=True,on_epoch=False,prog_bar=True)
        	self.log("grad_norm",grad_norm,on_step=True,on_epoch=False,prog_bar=True)
    
    def training_step_end(self, outputs):
        preds, labels = outputs['preds'], outputs['labels']
        data_time, model_time = outputs['data_time'], outputs['model_time']
        loss = self.loss_fn(preds, labels)
        self.train_top1_acc(preds.softmax(dim=-1), labels)
        self.train_top5_acc(preds.softmax(dim=-1), labels)
        
        self.log_step_state(data_time, model_time)
        self.iteration += 1
        self.data_start = time.perf_counter()
        return loss

    def training_epoch_end(self, outputs):
        mean_top1_acc = self.train_top1_acc.compute()
        mean_top5_acc = self.train_top5_acc.compute()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.print(f'{timestamp} - Evaluating mean ',
        		   f'top1_acc:{mean_top1_acc:.3f},',
        		   f'top5_acc:{mean_top5_acc:.3f} of current training epoch')
        self.train_top1_acc.reset()
        self.train_top5_acc.reset()

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
        	
        	# save checkpoint
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