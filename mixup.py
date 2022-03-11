""" Mixup and Cutmix

Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)

CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899)

Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch

Hacked together by / Copyright 2019, Ross Wightman
"""
import numpy as np
import torch

def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
	x = x.long().view(-1, 1)
	return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)

def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
	off_value = smoothing / num_classes
	on_value = 1. - smoothing + off_value
	y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
	y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
	return y1 * lam + y2 * (1. - lam)

def rand_bbox(img_shape, lam, margin=0., count=None):
	""" Standard CutMix bounding-box
	Generates a random square bbox based on lambda value. This impl includes
	support for enforcing a border margin as percent of bbox dimensions.

	Args:
		img_shape (tuple): Image shape as tuple
		lam (float): Cutmix lambda value
		margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
		count (int): Number of bbox to generate
	"""
	ratio = np.sqrt(1 - lam)
	img_h, img_w = img_shape[-2:]
	cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
	margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
	cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
	cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
	yl = np.clip(cy - cut_h // 2, 0, img_h)
	yh = np.clip(cy + cut_h // 2, 0, img_h)
	xl = np.clip(cx - cut_w // 2, 0, img_w)
	xh = np.clip(cx + cut_w // 2, 0, img_w)
	return yl, yh, xl, xh

def cutmix_bbox_and_lam(img_shape, lam, correct_lam=True, count=None):
	""" Generate bbox and apply lambda correction.
	"""
	yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
	if correct_lam:
		bbox_area = (yu - yl) * (xu - xl)
		lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
	return (yl, yu, xl, xu), lam

class Mixup:
	""" Mixup/Cutmix that applies different params to each element or whole batch

	Args:
		mixup_alpha (float): mixup alpha value, mixup is active if > 0.
		cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
		prob (float): probability of applying mixup or cutmix per batch or element
		switch_prob (float): probability of switching to cutmix instead of mixup when both are active
		mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
		correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
		label_smoothing (float): apply label smoothing to the mixed target tensor
		num_classes (int): number of classes for target
	"""
	def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
				 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
		self.mixup_alpha = mixup_alpha
		self.cutmix_alpha = cutmix_alpha
		self.mix_prob = prob
		self.switch_prob = switch_prob
		self.label_smoothing = label_smoothing
		self.num_classes = num_classes
		self.mode = mode
		self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
		self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

	def _params_per_batch(self):
		lam = 1.
		use_cutmix = False
		if self.mixup_enabled and np.random.rand() < self.mix_prob:
			if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
				use_cutmix = np.random.rand() < self.switch_prob
				lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
					np.random.beta(self.mixup_alpha, self.mixup_alpha)
			elif self.mixup_alpha > 0.:
				lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
			elif self.cutmix_alpha > 0.:
				use_cutmix = True
				lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
			else:
				assert False, "One of mixup_alpha > 0., cutmix_alpha > 0."
			lam = float(lam_mix)
		return lam, use_cutmix

	def _mix_batch(self, x):
		lam, use_cutmix = self._params_per_batch()
		if lam == 1.:
			return 1.
		if use_cutmix:
			(yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
				x.shape, lam, correct_lam=self.correct_lam)
			x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
		else:
			x_flipped = x.flip(0).mul_(1. - lam)
			x.mul_(lam).add_(x_flipped)
		return lam

	def __call__(self, x, target):
		assert len(x) % 2 == 0, 'Batch size should be even when using this' # [B,C,H,W] -> [B,T,C,H,W]
		need_reshape = False
		if x.ndim == 5:
			need_reshape = True	
			b,t,c,h,w = x.shape
			x = x.view(b,t*c,h,w)
		lam = self._mix_batch(x)
		target = mixup_target(target, self.num_classes, lam, self.label_smoothing, x.device)
		if need_reshape:
			x = x.view(b,t,c,h,w)
		return x, target
		
if __name__ == '__main__':
	SEED = 0
	torch.random.manual_seed(SEED)
	np.random.seed(SEED)
	mixupfn = Mixup(num_classes=4)
	x = torch.rand(2,2,1,10,10)
	label = [0, 1]
	print(x, label)
	y = torch.from_numpy(np.array(label))
	x, y = mixupfn(x, y)
	print(x.shape, y.shape)
	print(x, y)