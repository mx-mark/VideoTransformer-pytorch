import glob
import json
import os
import os.path as osp

import decord
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

class_labels_map = None
cls_sample_cnt = None


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
			img = denormalize(img, mean, std).cpu().numpy()
			img = (img * 255).astype(np.uint8)
			ax.imshow(np.asarray(img))
			ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

	plt.tight_layout()
	filename = osp.join(save_dir, f'clip_transformed_b{index}.png')
	plt.savefig(filename)
 
 
def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(num_class, anno_pth='./k600_classmap.json'):
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
			class_index = class_to_idx[class_name]
			
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
                 target_video_len=32,
                 align_transform=None,
                 temporal_sample=None):
        self.data = load_annotations(annotation_path, num_class, num_samples_per_cls)

        self.align_transform = align_transform
        self.temporal_sample = temporal_sample
        self.target_video_len = target_video_len

        self.v_decoder = DecordInit()

    def __getitem__(self, index):
        path = self.data[index]['video']
        try:
            v_reader = self.v_decoder(path)
        except:
            return None, None, -1
        
        # Sampling video frames
        total_frames = len(v_reader)
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        if end_frame_ind-start_frame_ind < self.target_video_len:
            return None, None, -1
        frame_indice = np.linspace(0, end_frame_ind-start_frame_ind-1, 
        						   self.target_video_len, dtype=int)
        video = v_reader.get_batch(frame_indice).asnumpy()
        del v_reader
        
        # Video align transform: T C H W
        with torch.no_grad():
        	video = torch.from_numpy(video).permute(0,3,1,2)
        	if self.align_transform is not None:
        		self.align_transform.randomize_parameters()
        		video = self.align_transform(video)
        
        # Label
        label = self.data[index]['label']
        
        return video, label

    def __len__(self):
        return len(self.data)