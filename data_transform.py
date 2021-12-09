from collections.abc import Sequence
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


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
        imgs = self._color_jit(imgs)
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