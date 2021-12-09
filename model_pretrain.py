import comet_ml

import os
import time
import random
import warnings
import argparse

import kornia.augmentation as K
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.utils.data as data

from data_trainer import KineticsDataModule, DataAugmentation
from model_trainer import VideoTransformer
import data_transform as T
from utils import print_on_rank_zero


def parse_args():
    parser = argparse.ArgumentParser(description='lr receiver')
    parser.add_argument(
        '-lr', type=float, required=True,
        help='the initial learning rate')
    parser.add_argument(
        '-epoch', type=int, required=True,
        help='the max epochs of training')
    parser.add_argument(
        '-gpus', nargs='+', type=int, default=-1,
        help='the avaiable gpus in this experiment')
    parser.add_argument(
        '-batch_size', type=int, required=True,
        help='the batch size of data inputs')
    parser.add_argument(
        '-num_workers', type=int, default=4,
        help='the num workers of loading data')
    parser.add_argument(
        '-log_interval', type=int, default=30,
        help='the intervals of logging')
    parser.add_argument(
        '-num_class', type=int, required=True,
        help='the num class of dataset used')
    parser.add_argument(
        '-num_samples_per_cls', type=int, default=10000,
        help='the num samples of per class')
    parser.add_argument(
        '-pretrain', type=str, default='vit',
        help='the pretrain params from [mae, vit]')
    parser.add_argument(
        '-frame_interval', type=int, required=True,
        help='the intervals of frame sampling')
    parser.add_argument(
        '-seed', type=int, default=0,
        help='the seed of exp')
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
        '-root_dir', type=str, required=True,
        help='the path to root dir for work space')
    args = parser.parse_args()
    
    return args

def single_run():
	args = parse_args()
	warnings.filterwarnings('ignore')
	
	# Experiment Settings
	MAX_EPOCHS = args.epoch
	BATCH_SIZE = args.batch_size
	NUM_WORKERS = args.num_workers
	ARCH = 'timesformer'
	SEED = args.seed
	N_VIDEO_FRAMES = 8
	IMG_SIZE = 224
	N_FRAME_INTERVAL = args.frame_interval
	
	ROOT_DIR = args.root_dir
	train_ann_path = args.train_data_path
	val_ann_path = args.val_data_path
	test_ann_path = args.test_data_path
	if 'mae' in args.pretrain:
		ckpt_pth = 'pretrain_model/pretrain_mae_vit_base_mask_0.75_400e.pth'
		pretrain_pth = os.path.join(ROOT_DIR, ckpt_pth)
	else:
		ckpt_pth = 'pretrain_model/vit_base_patch16_224.pth'
		pretrain_pth = os.path.join(ROOT_DIR, ckpt_pth)

	# TimeSformer-B settings	
	model_kwargs = {
		'pretrained':pretrain_pth,
		'num_frames':N_VIDEO_FRAMES,
		'img_size':IMG_SIZE,
		'patch_size':16,
		'embed_dims':768,
		'in_channels':3,
		'dropout_ratio':0.0,
		'attention_type':'divided_space_time',
		#common
		'n_crops':3,
		'lr':args.lr,
		'num_classes':args.num_class,
		'log_interval':args.log_interval,
		}
		
	exp_tag = (f'arch_{ARCH}_lr_{args.lr}_'
			  f'pretrain_{args.pretrain}_seed_{SEED}_'
			  f'gpus_{torch.cuda.device_count()}_'
			  f'bs_{BATCH_SIZE}_nw_{NUM_WORKERS}_'
			  f'frame_interval_{N_FRAME_INTERVAL}')
	ckpt_dir = os.path.join(ROOT_DIR, f'results/{exp_tag}/ckpt')
	log_dir = os.path.join(ROOT_DIR, f'results/{exp_tag}/log')
	os.makedirs(ckpt_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
    
	# Data
	mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
	norm_transform = K.Normalize(mean, std)
	# train
	train_align_transform = T.Compose([
		T.Resize(scale_range=(256, 320)),
		T.RandomCrop(size=IMG_SIZE),
		T.ToTensor(),
		])
	train_aug_transform = K.VideoSequential(
		K.RandomHorizontalFlip(p=0.5),
		data_format="BTCHW",
		same_on_frame=True,
		)
	train_aug_transform = DataAugmentation(norm_transform, train_aug_transform)
	train_temporal_sample = T.TemporalRandomCrop(N_VIDEO_FRAMES*N_FRAME_INTERVAL)
	
	# val
	if val_ann_path is not None:
		val_align_transform = T.Compose([
			T.Resize(scale_range=(-1, 256)),
			T.CenterCrop(size=IMG_SIZE),
			T.ToTensor(),
			])
		val_aug_transform = DataAugmentation(norm_transform)
		val_temporal_sample = T.TemporalRandomCrop(N_VIDEO_FRAMES*N_FRAME_INTERVAL)
		do_eval = True
	else:
		val_align_transform = None
		val_aug_transform = None
		val_temporal_sample = None
		do_eval = False
		
	# test
	if test_ann_path is not None:
		test_align_transform = T.Compose([
			T.Resize(scale_range=(-1, 224)),
			T.ThreeCrop(size=IMG_SIZE),
			T.ToTensor(),
			])
		test_aug_transform = DataAugmentation(norm_transform)
		test_temporal_sample = T.TemporalRandomCrop(N_VIDEO_FRAMES*N_FRAME_INTERVAL)
		do_test = True
	else:
		test_align_transform = None
		test_aug_transform = None
		test_temporal_sample = None
		do_test = False
	
	data_module = KineticsDataModule(
		train_ann_path=train_ann_path,
		train_align_transform=train_align_transform,
		train_aug_transform=train_aug_transform,
		train_temporal_sample=train_temporal_sample,
		val_ann_path=val_ann_path,
		val_align_transform=val_align_transform,
		val_aug_transform=val_aug_transform,
		val_temporal_sample=val_temporal_sample,
		test_ann_path=test_ann_path,
		test_align_transform=test_align_transform,
		test_aug_transform=test_aug_transform,
		test_temporal_sample=test_temporal_sample,
		num_class=args.num_class, 
        num_samples_per_cls=args.num_samples_per_cls,
		target_video_len=N_VIDEO_FRAMES,
		batch_size=BATCH_SIZE,
		num_workers=NUM_WORKERS)
	
	# Logger
	comet_logger = CometLogger(
		save_dir=log_dir,
		project_name="Pretrain on K600",
		experiment_name=exp_tag,
		offline=True)

	# Trainer
	trainer = pl.Trainer(
		gpus=args.gpus, # devices=-1
		accelerator="ddp", # accelerator="gpu",strategy='ddp'
		plugins=DDPPlugin(find_unused_parameters=False),
		max_epochs=MAX_EPOCHS,
		callbacks=[
			LearningRateMonitor("epoch"),
		],
		logger=comet_logger,
		check_val_every_n_epoch=1,
		log_every_n_steps=args.log_interval,
		progress_bar_refresh_rate=args.log_interval,
		flush_logs_every_n_steps=args.log_interval*5)
		
	# To be reproducable
	torch.random.manual_seed(SEED)
	np.random.seed(SEED)
	random.seed(SEED)
	pl.seed_everything(SEED, workers=True)
	
	# Model
	model = VideoTransformer(**model_kwargs, 
							 trainer=trainer,
							 ckpt_dir=ckpt_dir,
							 do_eval=do_eval,
							 do_test=do_test)
	
	timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
	print_on_rank_zero(f'{timestamp} - INFO - Start running,')
	trainer.fit(model, data_module)
	trainer.test(model, data_module)
    
if __name__ == '__main__':
	single_run()
