from collections.abc import Sequence
import random
import math

from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
_torch_interpolation_to_str = {
			InterpolationMode.NEAREST: 'nearest',
			InterpolationMode.BILINEAR: 'bilinear',
			InterpolationMode.BICUBIC: 'bicubic',
			InterpolationMode.BOX: 'box',
			InterpolationMode.HAMMING: 'hamming',
			InterpolationMode.LANCZOS: 'lanczos',
		}
_str_to_torch_interpolation = {b: a for a, b in _torch_interpolation_to_str.items()}

def str_to_interp_mode(mode_str):
	return _str_to_torch_interpolation[mode_str]

#  ------------------------------------------------------------
#  ----------------------  Common  ----------------------------
#  ------------------------------------------------------------
class Compose(object):
	"""Composes several transforms together.

	Args:
		transforms (list of transform objects): list of data transforms to compose.
	"""

	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, img):
		for t in self.transforms:
			img = t(img)
		return img

	def randomize_parameters(self):
		for t in self.transforms:
			if hasattr(t, 'randomize_parameters'):
				t.randomize_parameters()


class ToTensor(object):
	"""Convert a tensor to torch.FloatTensor in the range [0.0, 1.0].

	Args:
		norm_value (int): the max value of the input image tensor, default to 255.
	"""

	def __init__(self, norm_value=255):
		self.norm_value = norm_value

	def __call__(self, pic):
		if isinstance(pic, torch.Tensor):
			return pic.float().div(self.norm_value)

	def randomize_parameters(self):
		pass
	
	
#  ------------------------------------------------------------
#  -------------------  Transformation  -----------------------
#  ------------------------------------------------------------
class RandomCrop(object):
	"""Random crop a fixed size region in a given image.

	Args:
		size (int, Tuple[int]): Desired output size (out_h, out_w) of the crop
	"""

	def __init__(self, size):
		if isinstance(size, tuple):
			if size[0] != size[1]:
				raise ValueError(f'crop size {size[0], size[1]}, must be equal.')
			else:
				self.size = size[0]
		else:
			self.size = size

	def __call__(self, imgs):
		# Crop size
		size = self.size
	
		# Location
		img_height, img_width  = imgs.size(2), imgs.size(3)
		y_offset = int(self.y_jitter * (img_height - size))
		x_offset = int(self.x_jitter * (img_width - size))
	
		imgs = imgs[..., y_offset : y_offset + size, x_offset : x_offset + size]
		return imgs

	def __repr__(self):
		repr_str = (f'{self.__class__.__name__}('
					f'size={self.size})')
		return repr_str

	def randomize_parameters(self):
		self.x_jitter = random.random()
		self.y_jitter = random.random()


class Resize(object):
	"""Resize images to a specific size.

	Args:
		scale_range (Tuple[int]): If the first value equals to -1, the second value 
			serves as a short edge of the resized image: else if it is a tuple of 2 
			integers, the short edge of resized image will be random choice from
			[scale_range[0], scale_range[1]].
	"""

	def __init__(self, scale_range):
		if not isinstance(scale_range, tuple):
			raise ValueError(f'Scale_range {scale_range}, must be tuple.')
		self.scale_range = scale_range

	def __call__(self, imgs):
		imgs = self._resize(imgs)
		return imgs

	def __repr__(self):
		repr_str = (f'{self.__class__.__name__}('
					f'size={self.size})')
		return repr_str

	def randomize_parameters(self):
		if self.scale_range[0] == -1:
			self._resize = transforms.Resize(self.scale_range[1])
		else:
			short_edge = np.random.randint(self.scale_range[0],
										   self.scale_range[1]+1)
			self._resize = transforms.Resize(short_edge)


class RandomResizedCrop:
	"""Random crop that specifics the area and height-weight ratio range.

	Args:
		area_range (Tuple[float]): The candidate area scales range of
			output cropped images. Default: (0.08, 1.0).
		aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
			output cropped images. Default: (3 / 4, 4 / 3).
	"""

	def __init__(self,
				 size,
				 interpolation=3,
				 scale=(0.08, 1.0),
				 ratio=(3 / 4, 4 / 3)):
		self.size = size
		self.area_range = scale
		self.aspect_ratio_range = ratio
		self.interpolation = interpolation
	
	def __call__(self, imgs):
		"""Performs the RandomResizeCrop augmentation.

		Args:
			results (dict): The resulting dict to be modified and passed
				to the next transform in pipeline.
		"""
		# version one- frame diverse
		#imgs = self._crop_imgs(imgs)
	
		# version two- frame consistent
		img_width = imgs.shape[-1]
		img_height = imgs.shape[-2]
		# crop size
		min_length = min(img_width, img_height)
		crop_size = int(min_length * self.scale)
		width = crop_size
		height = crop_size*self.ratio

		# location
		left = self.tl_x * (img_width - width)
		top = self.tl_y * (img_height - height)
	
		imgs = transforms.functional.resized_crop(
			imgs, int(top), int(left), int(height), int(width), self.size, interpolation=self.interpolation)

		return imgs

	def __repr__(self):
		repr_str = (f'{self.__class__.__name__}('
					f'area_range={self.area_range}, '
					f'aspect_ratio_range={self.aspect_ratio_range}, '
					f'size={self.size})')
		return repr_str

	def randomize_parameters(self):
		self.scale = random.uniform(self.area_range[0], self.area_range[1])
		self.ratio = random.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1])
		'''
		# version one- frame diverse
		self._crop_imgs = transforms.RandomResizedCrop(
			self.size, scale=(scale, scale), ratio=(ratio, ratio))
		'''
		# version two- frame consistent
		self.tl_x = random.random()
		self.tl_y = random.random()  


class Flip(object):
	"""Flip the input images with a probability.

	Args:
		flip_ratio (float): Probability of implementing flip. Default: 0.5.
	"""

	def __init__(self,
				 flip_ratio=0.5):
		self.flip_ratio = flip_ratio

	def __call__(self, imgs):
		imgs = self._flip(imgs)
		return imgs

	def __repr__(self):
		repr_str = (
			f'{self.__class__.__name__}('
			f'flip_ratio={self.flip_ratio})')
		return repr_str

	def randomize_parameters(self):
		p = random.random()
		if p > self.flip_ratio:
			self._flip = transforms.RandomHorizontalFlip(p=1)
		else:
			self._flip = transforms.RandomHorizontalFlip(p=0)


class RandomGrayscale(object):
	"""Flip the input images with a probability.

	Args:
		flip_ratio (float): Probability of implementing flip. Default: 0.5.
	"""

	def __init__(self,
				 p=0.1):
		self.p = p

	def __call__(self, imgs):
		imgs = self._grayscale(imgs)
		return imgs

	def __repr__(self):
		repr_str = (
			f'{self.__class__.__name__}('
			f'p={self.p})')
		return repr_str

	def randomize_parameters(self):
		p = random.random()
		if p > self.p:
			self._grayscale = transforms.RandomGrayscale(p=0)
		else:
			self._grayscale = transforms.RandomGrayscale(p=1)


class RandomApply(object):
	"""Flip the input images with a probability.

	Args:
		flip_ratio (float): Probability of implementing flip. Default: 0.5.
	"""

	def __init__(self,
				 transform,
				 p=0.5):
		self.p = p
		self.transform = transform

	def __call__(self, imgs):
		imgs = self._random_apply(imgs)
		return imgs

	def __repr__(self):
		repr_str = (
			f'{self.__class__.__name__}('
			f'p={self.p})')
		return repr_str

	def randomize_parameters(self):
		p = random.random()
		if p > self.p:
			self._random_apply = transforms.RandomApply(self.transform, p=0)
		else:
			self._random_apply = transforms.RandomApply(self.transform, p=1)


class Normalize(object):
	"""Normalize the images with the given mean and std value.

	Args:
		mean (Sequence[float]): Mean values of different channels.
		std (Sequence[float]): Std values of different channels.
	"""

	def __init__(self, mean, std):
		if not isinstance(mean, Sequence):
			raise TypeError(
				f'Mean must be list, tuple or np.ndarray, but got {type(mean)}'
			)

		if not isinstance(std, Sequence):
			raise TypeError(
				f'Std must be list, tuple or np.ndarray, but got {type(std)}')
			
		self._normalize = transforms.Normalize(mean, std)
		self.mean = mean
		self.std = std

	#@profile
	def __call__(self, imgs):
		imgs = self._normalize(imgs)
		return imgs

	def __repr__(self):
		repr_str = (f'{self.__class__.__name__}('
					f'mean={self.mean}, '
					f'std={self.std})')
		return repr_str

	def randomize_parameters(self):
		pass


class ColorJitter(object):
	"""Randomly distort the brightness, contrast, saturation and hue of images.

	Note: The input images should be in RGB channel order.

	Args:
		brightness (float): the std values of brightness distortion.
		contrast (float): the std values of contrast distortion.
		saturation (float): the std values of saturation distortion.
		hue (float): the std values of hue distortion.
	"""

	def __init__(self,
				 brightness=0, 
				 contrast=0, 
				 saturation=0, 
				 hue=0):
		self.brightness = brightness
		self.contrast = contrast
		self.saturation = saturation
		self.hue = hue

	def __call__(self, imgs):
		print(imgs.shape)
		if imgs.ndim == 3:
			imgs = rearrange(imgs, '(t c) h w -> t c h w', c=3)
			imgs = self._color_jit(imgs)
			imgs = rearrange(imgs, 't c h w -> (t c) h w')
		return imgs

	def __repr__(self):
		repr_str = (f'{self.__class__.__name__}('
					f'brightness={self.brightness}, '
					f'contrast={self.contrast}, '
					f'saturation={self.saturation}, '
					f'hue={self.hue})')
		return repr_str

	def randomize_parameters(self):
		brightness = random.uniform(max(0,1-self.brightness), 1+self.brightness)
		contrast = random.uniform(max(0,1-self.contrast), 1+self.contrast)
		saturation = random.uniform(max(0,1-self.saturation), 1+self.saturation)
		hue = random.uniform(-self.hue, self.hue)
	
		self._color_jit = transforms.ColorJitter(
			brightness=(brightness,brightness),
			contrast=(contrast,contrast),
			saturation=(saturation,saturation),
			hue=(hue,hue))


class CenterCrop(object):
	"""Crop the center area from images.

	Args:
		crop_size (int | tuple[int]): (w, h) of crop size.
	"""

	def __init__(self, size):
		self.size = size
		self._center_crop = transforms.CenterCrop(size=size)

	def __call__(self, imgs):
		imgs = self._center_crop(imgs)
		return imgs

	def __repr__(self):
		repr_str = (f'{self.__class__.__name__}(size={self.size})')
		return repr_str

	def randomize_parameters(self):
		pass


class ThreeCrop(object):
	"""Random crop the three pre-define regions of image.

	Args:
		size (int, Tuple[int]): Desired output size (out_h, out_w) of the crop
	"""

	def __init__(self, size):
		if isinstance(size, tuple):
			if size[0] != size[1]:
				raise ValueError(f'crop size {size[0], size[1]}, must be equal.')
			else:
				self.size = size[0]
		else:
			self.size = size

	def __call__(self, imgs):
		# Crop size
		size = int(self.size)
		img_height, img_width  = imgs.size(2), imgs.size(3)
		if size > img_height or size > img_width:
			msg = "Requested crop size {} is bigger than input size {}"
			raise ValueError(msg.format(size, (img_height, img_width)))
	
		# Location
		crops = []
		left_y_offset = (img_height - size) // 2
		left_x_offset = 0
		left = imgs[...,
					left_y_offset : left_y_offset + size,
					left_x_offset : left_x_offset + size]
		crops.append(left)
	
		right_y_offset = (img_height - size) // 2
		right_x_offset = img_width - size
		right = imgs[...,
					 right_y_offset : right_y_offset + size,
					 right_x_offset : right_x_offset + size]
		crops.append(right)
	
		center_y_offset = (img_height - size) // 2
		center_x_offset = (img_width - size) // 2
		center = imgs[...,
					  center_y_offset : center_y_offset + size,
					  center_x_offset : center_x_offset + size]
		crops.append(center)
	
		# (N_Crops T C H W)
		imgs = torch.stack(crops)
		return imgs

	def __repr__(self):
		repr_str = (f'{self.__class__.__name__}('
					f'size={self.size})')
		return repr_str

	def randomize_parameters(self):
		pass


#  ------------------------------------------------------------
#  ---------------------  Sampling  ---------------------------
#  ------------------------------------------------------------
class TemporalRandomCrop(object):
	"""Temporally crop the given frame indices at a random location.

	Args:
		size (int): Desired length of frames will be seen in the model.
	"""

	def __init__(self, size):
		self.size = size

	def __call__(self, total_frames):
		rand_end = max(0, total_frames - self.size - 1)
		begin_index = random.randint(0, rand_end)
		end_index = min(begin_index + self.size, total_frames)
		return begin_index, end_index


#  ------------------------------------------------------------
#  ---------------------  AdvancedAugment  --------------------
#  ------------------------------------------------------------
def transforms_train(img_size=224,
					 scale=None,
					 ratio=None,
					 hflip=0.5,
					 color_jitter=0.4,
					 auto_augment=None,
					 interpolation='random',
					 mean=IMAGENET_DEFAULT_MEAN,
					 std=IMAGENET_DEFAULT_STD,
					 objective='supervised'):
	"""
	If separate==True, the transforms are returned as a tuple of 3 separate transforms
	for use in a mixing dataset that passes
	 * all data through the first (primary) transform, called the 'clean' data
	 * a portion of the data through the secondary transform
	 * normalizes and converts the branches above with the third, final transform
	"""
	scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
	ratio = tuple(ratio or (3./4., 4./3.))  # default imagenet ratio range
	primary_tfl = [
		transforms.RandomResizedCrop(img_size, scale=scale, ratio=ratio, interpolation=str_to_interp_mode(interpolation))]
	if hflip > 0.:
		primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]

	secondary_tfl = []
	if auto_augment:
		secondary_tfl += [transforms.autoaugment.RandAugment()]
	elif color_jitter is not None:
		# color jitter is enabled when not using AA
		if isinstance(color_jitter, (list, tuple)):
			# color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
			# or 4 if also augmenting hue
			assert len(color_jitter) in (3, 4)
		else:
			# if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
			color_jitter = (float(color_jitter),) * 3
		secondary_tfl += [transforms.ColorJitter(*color_jitter)]

	final_tfl = []
	final_tfl += [
		ToTensor(),
		transforms.Normalize(
			mean=torch.tensor(mean),
			std=torch.tensor(std))
	]
	if objective == 'mim':
		return [Compose(primary_tfl + secondary_tfl), Compose(final_tfl)]
	else:
		return Compose(primary_tfl + secondary_tfl + final_tfl)


def transforms_eval(img_size=224,
					crop_pct=None,
					interpolation='bilinear',
					mean=IMAGENET_DEFAULT_MEAN,
					std=IMAGENET_DEFAULT_STD):
	crop_pct = crop_pct or DEFAULT_CROP_PCT

	if isinstance(img_size, (tuple, list)):
		assert len(img_size) == 2
		if img_size[-1] == img_size[-2]:
			# fall-back to older behaviour so Resize scales to shortest edge if target is square
			scale_size = int(math.floor(img_size[0] / crop_pct))
		else:
			scale_size = tuple([int(x / crop_pct) for x in img_size])
	else:
		scale_size = int(math.floor(img_size / crop_pct))

	tfl = [
		transforms.Resize(scale_size, interpolation=str_to_interp_mode(interpolation)),
		transforms.CenterCrop(img_size),
	]
	tfl += [
		ToTensor(),
		transforms.Normalize(
				 mean=torch.tensor(mean),
				 std=torch.tensor(std))
	]

	return Compose(tfl)


def create_video_transform(input_size=224,
						   is_training=False,
						   scale=None,
						   ratio=None,
						   hflip=0.5,
						   color_jitter=0.4,
						   auto_augment=None,
						   interpolation='bilinear',
						   mean=IMAGENET_DEFAULT_MEAN,
						   std=IMAGENET_DEFAULT_STD,
						   objective='supervised',
						   crop_pct=None):

	if isinstance(input_size, (tuple, list)):
		img_size = input_size[-2:]
	else:
		img_size = input_size

	if is_training:
		transform = transforms_train(
			img_size,
			scale=scale,
			ratio=ratio,
			hflip=hflip,
			color_jitter=color_jitter,
			auto_augment=auto_augment,
			interpolation=interpolation,
			mean=mean,
			std=std,
			objective=objective)
	else:
		transform = transforms_eval(
			img_size,
			interpolation=interpolation,
			mean=mean,
			std=std,
			crop_pct=crop_pct)

	return transform
