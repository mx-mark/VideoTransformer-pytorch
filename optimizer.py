# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by MX
# --------------------------------------------------------
from functools import partial
from torch import optim as optim

from utils import print_on_rank_zero


def build_optimizer(hparams, model, is_pretrain):
	if is_pretrain:
		return build_pretrain_optimizer(hparams, model)
	else:
		return build_finetune_optimizer(hparams, model)


def build_pretrain_optimizer(hparams, model):
	skip = {}
	skip_keywords = {}
	if hasattr(model, 'no_weight_decay'):
		skip = model.no_weight_decay()
	if hasattr(model, 'no_weight_decay_keywords'):
		skip_keywords = model.no_weight_decay_keywords()

	parameters = get_pretrain_param_groups(model, skip, skip_keywords)

	opt_lower = hparams.optim_type.lower()
	optimizer = None
	if opt_lower == 'sgd':
		optimizer = optim.SGD(parameters, momentum=0.9, nesterov=True,
							  lr=hparams.lr, weight_decay=hparams.weight_decay)
	elif opt_lower == 'adamw':
		optimizer = optim.AdamW(parameters, betas=(0.9, 0.999),
								lr=hparams.lr, weight_decay=hparams.weight_decay)

	return optimizer


def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
	has_decay = []
	no_decay = []
	has_decay_name = []
	no_decay_name = []

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
				check_keywords_in_name(name, skip_keywords):
			no_decay.append(param)
			no_decay_name.append(name)
		else:
			has_decay.append(param)
			has_decay_name.append(name)

	print_on_rank_zero(f'params_no_decay_name: {no_decay_name} \n params_decay_name: {has_decay_name}')
	return [{'params': no_decay, 'weight_decay': 0.},
			{'params': has_decay},]


def build_finetune_optimizer(hparams, model):
	if hparams.arch == 'mvit':
		if hparams.layer_decay == 1:
			get_layer_func = None
			scales = None
		else:
			num_layers = 16
			get_layer_func = partial(get_mvit_layer, num_layers=num_layers + 2)
			scales = list(hparams.layer_decay ** i for i in reversed(range(num_layers + 2))) #layer_decay=1 disable
	else:
		return build_pretrain_optimizer(hparams, model)

	skip = {}
	skip_keywords = {}
	if hasattr(model, 'no_weight_decay'):
		skip = model.no_weight_decay()
	if hasattr(model, 'no_weight_decay_keywords'):
		skip_keywords = model.no_weight_decay_keywords()

	parameters = get_finetune_param_groups(
		model, hparams.lr, hparams.weight_decay,
		get_layer_func, scales, skip, skip_keywords)

	opt_lower = hparams.optim_type.lower()
	optimizer = None
	if opt_lower == 'sgd':
		optimizer = optim.SGD(parameters, momentum=0.9, nesterov=True,
							  lr=hparams.lr, weight_decay=hparams.weight_decay)
	elif opt_lower == 'adamw':
		optimizer = optim.AdamW(parameters, betas=(0.9, 0.999),
								lr=hparams.lr, weight_decay=hparams.weight_decay)

	return optimizer


def get_mvit_layer(name, num_layers):
	layer_name = name.replace('mvit.', '')
	layer_name = layer_name.replace('model.', '')
	if layer_name in ("mask_token"):
		return 0
	elif layer_name.startswith("patch_embed") or layer_name.startswith('cls_positional_encoding'):
		return 0
	elif layer_name.startswith("blocks"):
		layer_id = int(layer_name.split('.')[1])
		return layer_id + 1
	else:
		return num_layers - 1


def get_finetune_param_groups(model, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
	parameter_group_names = {}
	parameter_group_vars = {}

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
				check_keywords_in_name(name, skip_keywords):
			group_name = "no_decay"
			this_weight_decay = 0.
		else:
			group_name = "decay"
			this_weight_decay = weight_decay
		if get_layer_func is not None:
			layer_id = get_layer_func(name)
			group_name = "layer_%d_%s" % (layer_id, group_name)
			#print(name, group_name)
		else:
			layer_id = None

		if group_name not in parameter_group_names:
			if scales is not None:
				scale = scales[layer_id]
			else:
				scale = 1.

			parameter_group_names[group_name] = {
				"group_name": group_name,
				"weight_decay": this_weight_decay,
				"params": [],
				"lr": lr * scale,
				"lr_scale": scale,
			}
			parameter_group_vars[group_name] = {
				"group_name": group_name,
				"weight_decay": this_weight_decay,
				"params": [],
				"lr": lr * scale,
				"lr_scale": scale
			}

		parameter_group_vars[group_name]["params"].append(param)
		parameter_group_names[group_name]["params"].append(name)
	return list(parameter_group_vars.values())


def check_keywords_in_name(name, keywords=()):
	isin = False
	for keyword in keywords:
		if keyword in name:
			isin = True
	return isin