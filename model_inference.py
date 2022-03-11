import argparse
from einops import rearrange, reduce, repeat
import numpy as np
import torch
import torch.nn as nn

from dataset import DecordInit, load_annotation_data
import data_transform as T
from transformer import ClassificationHead
from video_transformer import TimeSformer
from weight_init import init_from_kinetics_pretrain_

def parse_args():
    parser = argparse.ArgumentParser(description='lr receiver')
    parser.add_argument(
        '-pretrain', type=str, required=True,
        help='the path to pretrain model')
    parser.add_argument(
        '-class_map', type=str, default='./k600_classmap.json',
        help='the path to k600 class map')
    parser.add_argument(
        '-video_path', type=str, required=True,
        help='the path to the test video sample')
    parser.add_argument(
        '-num_frames', type=int, required=True,
        help='the number of frames will be processed by model')
    parser.add_argument(
        '-frame_interval', type=int, required=True,
        help='the intervals of frame sampling')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
	args = parse_args()
	# Load pre-trained model
	pretrain_pth = args.pretrain
	num_frames = args.num_frames
	frame_interval = args.frame_interval
	model = TimeSformer(num_frames=num_frames,
						pretrained=pretrain_pth,
						img_size=224,
						patch_size=16,
						embed_dims=768,
						in_channels=3,
						attention_type='divided_space_time',
						use_learnable_pos_emb=True,
						return_cls_token=True)
	cls_head = ClassificationHead(num_classes=600, in_channels=768)
	init_from_kinetics_pretrain_(cls_head, pretrain_pth, init_module='cls_head')
	model.eval()
	cls_head.eval()
	
	# Prepare data preprocess
	video_path = args.video_path
	video_decoder = DecordInit()
	v_reader = video_decoder(video_path)
	mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
	data_transform = T.Compose([
			T.Resize(scale_range=(-1, 224)),
			T.ThreeCrop(size=224),
			T.ToTensor(),
			T.Normalize(mean, std)
			])
	temporal_sample = T.TemporalRandomCrop(num_frames*frame_interval)
	
	# Sampling video frames
	total_frames = len(v_reader)
	start_frame_ind, end_frame_ind = temporal_sample(total_frames)
	if end_frame_ind-start_frame_ind < num_frames:
		raise ValueError(f'the total frames of the video {video_path} is less than {num_frames}')
	frame_indice = np.linspace(0, end_frame_ind-start_frame_ind-1, num_frames, dtype=int)
	video = v_reader.get_batch(frame_indice).asnumpy()
	del v_reader
	
	# Video transform: T C H W
	with torch.no_grad():
		video = torch.from_numpy(video).permute(0,3,1,2)
		if data_transform is not None:
			data_transform.randomize_parameters()
			video = data_transform(video)
		
		logits = model(video)
		output = cls_head(logits)
		output = output.view(3, 600).mean(0)
		cls_pred = output.argmax()

	# Predict class label
	class_map = load_annotation_data(args.class_map)
	for key, value in class_map.items():
		if value == cls_pred:
			print(f'the shape of ouptut: {output.shape}, and the prediction is: {key}')
			break