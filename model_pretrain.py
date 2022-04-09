import os
import time
import random
import warnings
import argparse

import kornia.augmentation as K
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.utils.data as data

from data_trainer import KineticsDataModule
from model_trainer import VideoTransformer
import data_transform as T
from utils import print_on_rank_zero


def parse_args():
	parser = argparse.ArgumentParser(description='lr receiver')
	# Common
	parser.add_argument(
		'-epoch', type=int, required=True,
		help='the max epochs of training')
	parser.add_argument(
		'-batch_size', type=int, required=True,
		help='the batch size of data inputs')
	parser.add_argument(
		'-num_workers', type=int, default=4,
		help='the num workers of loading data')
	parser.add_argument(
		'-resume', default=False, action='store_true')
	parser.add_argument(
		'-resume_from_checkpoint', type=str, default=None,
		help='the pretrain params from specific path')
	parser.add_argument(
		'-log_interval', type=int, default=30,
		help='the intervals of logging')
	parser.add_argument(
		'-save_ckpt_freq', type=int, default=20,
		help='the intervals of saving model')
	parser.add_argument(
		'-objective', type=str, default='mim',
		help='the learning objective from [mim, supervised]')
	parser.add_argument(
		'-eval_metrics', type=str, default='finetune',
		help='the eval metrics choosen from [linear_prob, finetune]')

	# Environment
	parser.add_argument(
		'-gpus', nargs='+', type=int, default=-1,
		help='the avaiable gpus in this experiment')
	parser.add_argument(
		'-root_dir', type=str, required=True,
		help='the path to root dir for work space')

	# Data
	parser.add_argument(
		'-num_class', type=int, required=True,
		help='the num class of dataset used')
	parser.add_argument(
		'-num_samples_per_cls', type=int, default=10000,
		help='the num samples of per class')
	parser.add_argument(
		'-img_size', type=int, default=224,
		help='the size of processed image')
	parser.add_argument(
		'-num_frames', type=int, required=True,
		help='the mumber of frame sampling')
	parser.add_argument(
		'-frame_interval', type=int, required=True,
		help='the intervals of frame sampling')
	parser.add_argument(
		'-data_statics', type=str, default='kinetics',
		help='choose data statics from [imagenet, kinetics]')
	parser.add_argument(
		'-train_data_path', type=str, required=True,
		help='the path to train set')
	parser.add_argument(
		'-val_data_path', type=str, default=None,
		help='the path to val set')
	parser.add_argument(
		'-test_data_path', type=str, default=None,
		help='the path to test set')
	parser.add_argument(
		'-multi_crop', type=bool, default=False, 
		help="""Whether or not to use multi crop.""")
	parser.add_argument(
		'-mixup', type=bool, default=False, 
		help="""Whether or not to use multi crop.""")
	parser.add_argument(
		'-auto_augment', type=str, default=None,
		help='the used Autoaugment policy')
		
	# Model
	parser.add_argument(
		'-arch', type=str, default='timesformer',
		help='the choosen model arch from [timesformer, vivit]')
	parser.add_argument(
		'-attention_type', type=str, default='divided_space_time',
		help='the choosen attention type using in model')
	parser.add_argument(
		'-pretrain_pth', type=str, default=None,
		help='the path to the pretrain weights')
	parser.add_argument(
		'-weights_from', type=str, default='imagenet',
		help='the pretrain params from [imagenet, kinetics]')

	# Training/Optimization parameters
	parser.add_argument(
		'-seed', type=int, default=0,
		help='the seed of exp')
	parser.add_argument(
		'-optim_type', type=str, default='adamw',
		help='the optimizer using in the training')
	parser.add_argument(
		'-lr_schedule', type=str, default='cosine',
		help='the lr schedule using in the training')
	parser.add_argument(
		'-lr', type=float, required=True,
		help='the initial learning rate')
	parser.add_argument(
		'-layer_decay', type=float, default=0.75,
		help='the value of layer_decay')
	parser.add_argument(
		'--min_lr', type=float, default=1e-6, 
		help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""")
	parser.add_argument(
		'-use_fp16', type=bool, default=True, 
		help="""Whether or not to use half precision for training. Improves training time and memory requirements,
		but can provoke instability and slight decay of performance. We recommend disabling
		mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
	parser.add_argument(
		'-weight_decay', type=float, default=0.05, 
		help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""")
	parser.add_argument(
		'-weight_decay_end', type=float, default=0.05, 
		help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by
		the end of training improves performance for ViTs.""")
	parser.add_argument(
		'-clip_grad', type=float, default=0, 
		help="""Maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
		help optimization for larger ViT architectures. 0 for disabling.""")
	parser.add_argument(
		"-warmup_epochs", default=5, type=int,
		help="Number of epochs for the linear learning-rate warm up.")

	args = parser.parse_args()
	
	return args

def single_run():
	args = parse_args()
	warnings.filterwarnings('ignore')
	
	# linear learning rate scale
	if isinstance(args.gpus, int):
		num_gpus = torch.cuda.device_count()
	else:
		num_gpus = len(args.gpus)
	effective_batch_size = args.batch_size * num_gpus
	args.lr = args.lr * effective_batch_size / 256

	# Experiment Settings
	ROOT_DIR = args.root_dir
	exp_tag = (f'objective_{args.objective}_arch_{args.arch}_lr_{args.lr}_'
			   f'optim_{args.optim_type}_lr_schedule_{args.lr_schedule}_'
			   f'fp16_{args.use_fp16}_weight_decay_{args.weight_decay}_'
			   f'weight_decay_end_{args.weight_decay_end}_warmup_epochs_{args.warmup_epochs}_'
			   f'pretrain_{args.pretrain_pth}_weights_from_{args.weights_from}_seed_{args.seed}_'
			   f'img_size_{args.img_size}_num_frames_{args.num_frames}_eval_metrics_{args.eval_metrics}_'
			   f'frame_interval_{args.frame_interval}_mixup_{args.mixup}_'
			   f'multi_crop_{args.multi_crop}_auto_augment_{args.auto_augment}_')
	ckpt_dir = os.path.join(ROOT_DIR, f'results/{exp_tag}/ckpt')
	log_dir = os.path.join(ROOT_DIR, f'results/{exp_tag}/log')
	os.makedirs(ckpt_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)

	# Data
	do_eval = True if args.val_data_path is not None else False
	do_test = True if args.test_data_path is not None else False
	
	data_module = KineticsDataModule(configs=args,
									 train_ann_path=args.train_data_path,
									 val_ann_path=args.val_data_path,
									 test_ann_path=args.test_data_path)
	
	# Resume from the last checkpoint
	if args.resume and not args.resume_from_checkpoint:
		args.resume_from_checkpoint = os.path.join(ckpt_dir, 'last_checkpoint.pth')

	# Trainer
	if args.arch == 'mvit' and args.objective == 'supervised':
		find_unused_parameters = True
	else:
		find_unused_parameters = False

	trainer = pl.Trainer(
		gpus=args.gpus, 
		accelerator="ddp",
		precision=16,
		plugins=[DDPPlugin(find_unused_parameters=find_unused_parameters),],
		max_epochs=args.epoch,
		callbacks=[
			LearningRateMonitor(logging_interval='step'),
		],
		resume_from_checkpoint=args.resume_from_checkpoint,
		check_val_every_n_epoch=1,
		log_every_n_steps=args.log_interval,
		progress_bar_refresh_rate=args.log_interval,
		flush_logs_every_n_steps=args.log_interval*5)
		
	# To be reproducable
	torch.random.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	pl.seed_everything(args.seed, workers=True)
	
	# Model
	model = VideoTransformer(configs=args, 
							 trainer=trainer,
							 ckpt_dir=ckpt_dir,
							 do_eval=do_eval,
							 do_test=do_test)
	print_on_rank_zero(args)
	timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
	print_on_rank_zero(f'{timestamp} - INFO - Start running,')
	trainer.fit(model, data_module)
	
if __name__ == '__main__':
	single_run()