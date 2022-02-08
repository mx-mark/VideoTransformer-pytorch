import math
import random
import numpy as np

class RandomMaskGenerator:
	def __init__(self, input_size=224, mask_ratio=0.6):
		if not isinstance(input_size, tuple):
			input_size = (input_size,) * 2

		self.height, self.width = input_size

		self.num_patches = self.height * self.width
		self.num_mask = int(mask_ratio * self.num_patches)
		
	def __call__(self):
		mask = np.hstack([
			np.zeros(self.num_patches - self.num_mask),
			np.ones(self.num_mask),
		])
		np.random.shuffle(mask) #
		return mask # [1024] 

class CubeMaskGenerator:
	def __init__(
			self, input_size=(8,14,14), mask_ratio=0.4, min_num_patches=16, max_num_patches=None,
			min_aspect=0.3, max_aspect=None):
		self.temporal ,self.height, self.width = input_size

		self.num_patches = self.height * self.width
		self.num_masking_patches = int(self.num_patches * mask_ratio)
		self.num_masking_frames = int(self.temporal * mask_ratio)

		self.min_num_patches = min_num_patches # smaller than max_num_patches
		self.max_num_patches = self.num_masking_patches if max_num_patches is None else max_num_patches

		max_aspect = max_aspect or 1 / min_aspect
		self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

	def __repr__(self):
		repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
			self.height, self.width, self.min_num_patches, self.max_num_patches,
			self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
		return repr_str

	def get_shape(self):
		return self.temporal, self.height, self.width

	def _mask(self, mask, max_mask_patches):
		delta = 0
		for attempt in range(10):
			target_area = random.uniform(self.min_num_patches, max_mask_patches)
			aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
			h = int(round(math.sqrt(target_area * aspect_ratio)))
			w = int(round(math.sqrt(target_area / aspect_ratio)))
			if w < self.width and h < self.height:
				top = random.randint(0, self.height - h)
				left = random.randint(0, self.width - w)

				num_masked = mask[top: top + h, left: left + w].sum()
				# Overlap
				if 0 < h * w - num_masked <= max_mask_patches:
					for i in range(top, top + h):
						for j in range(left, left + w):
							if mask[i, j] == 0:
								mask[i, j] = 1
								delta += 1

				if delta > 0:
					break
		return delta

	def __call__(self):
		time_marker = np.zeros(shape=self.temporal, dtype=np.int32)
		cube_mask = np.zeros(shape=self.get_shape(), dtype=np.int32)
		cube_marker = []
		temp_mask_count = 0
		while temp_mask_count < self.num_masking_frames:
			# generate 2D block-wise mask
			mask = np.zeros(shape=self.get_shape()[1:], dtype=np.int32)
			mask_count = 0
			while mask_count < self.num_masking_patches:
				max_mask_patches = self.num_masking_patches - mask_count
				max_mask_patches = min(max_mask_patches, self.max_num_patches)

				delta = self._mask(mask, max_mask_patches)
				if delta == 0:
					break
				else:
					mask_count += delta
			# assign to cube mask
			start_frame = random.randint(0, self.temporal)
			accumulate_frames = random.randint(1, self.num_masking_frames - temp_mask_count)
			mask_count = 0
			for i in range(start_frame, start_frame+accumulate_frames):
				if i > self.temporal-1:
					break
				if time_marker[i] == 0: # only update the unmask frame
					time_marker[i] = 1
					cube_mask[i] = mask
					mask_count+=1
				else: #avoid to overlap the orginal mask
					break
			temp_mask_count += mask_count
			if mask_count > 0: # mark the center frame index(mask_count > 0)
				cube_marker.append([start_frame, mask_count])
		
		return cube_mask, cube_marker

if __name__ == '__main__':
		# Unit test for computing cube mask and extracting hog features
		
		'''
		mask_generator = CubeMaskGenerator(input_size=(8,14,14),min_num_patches=16)
		mask, cube_marker = mask_generator()
		print(mask)
		from einops import repeat
		mask = repeat(mask, 't h w -> t (h dh) (w dw)', dh=56//14, dw=56//14) # nearest-neighbor resize
		print(mask)
		
		#print(cube_marker)
		center_index = np.zeros(8).astype('bool')
		for marker in cube_marker:
			center_index[marker[0]+marker[1]//2] = 1
		mask[~center_index] = 0
		print(mask)
		#for i in cube_marker:
			#print(mask[i].sum()/(56*56))
		'''

		#'''
		from skimage.feature import hog
		from skimage import io
		from skimage import data
		from einops import rearrange
		import torch
		
		def extract_hog(image):
			hog_features_r = hog(image[:,:,0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2', feature_vector=False)
			hog_features_g = hog(image[:,:,1], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2', feature_vector=False)
			hog_features_b, hog_image = hog(image[:,:,2], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2', feature_vector=False, visualize=True) 
			hog_features = np.concatenate([hog_features_r,hog_features_g,hog_features_b], axis=-1)
			hog_features = rearrange(hog_features, '(ph dh) (pw dw) ch cw c -> ph pw (dh dw ch cw c)', ph=14, pw=14)
			return hog_features
		
		images = np.zeros((2,224,224,3))
		image = data.astronaut()[:224,:224,:] # h w c
		image = torch.from_numpy(image).numpy()
		images[0] = image
		image = io.imread('./test_1.jpg')[:224,:224,:]
		images[1] = image
		hog_features = np.stack(list(map(extract_hog, images)), axis=0)
		print(hog_features.shape, np.min(hog_features), np.max(hog_features))
		#io.imsave('./test_img_hog.jpg',hog_image)
		#'''