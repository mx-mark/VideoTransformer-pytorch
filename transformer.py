from einops import rearrange, repeat, reduce
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from weight_init import trunc_normal_, constant_init_, kaiming_init_


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sine_cosine_pos_emb(n_position, d_hid): 
	''' Sinusoid position encoding table ''' 
	# TODO: make it with torch instead of numpy 
	def get_position_angle_vec(position): 
		return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

	sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
	sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
	sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

	return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DropPath(nn.Module):
	
	def __init__(self, dropout_p=None):
		super(DropPath, self).__init__()
		self.dropout_p = dropout_p

	def forward(self, x):
		return self.drop_path(x, self.dropout_p, self.training)
	
	def drop_path(self, x, dropout_p=0., training=False):
		if dropout_p == 0. or not training:
			return x
		keep_prob = 1 - dropout_p
		shape = (x.shape[0],) + (1,) * (x.ndim - 1)
		random_tensor = keep_prob + torch.rand(shape).type_as(x)
		random_tensor.floor_()  # binarize
		output = x.div(keep_prob) * random_tensor
		return output


class ClassificationHead(nn.Module):
	"""Classification head for Video Transformer.
	
	Args:
		num_classes (int): Number of classes to be classified.
		in_channels (int): Number of channels in input feature.
		init_std (float): Std value for Initiation. Defaults to 0.02.
		kwargs (dict, optional): Any keyword argument to be used to initialize
			the head.
	"""

	def __init__(self,
				 num_classes,
				 in_channels,
				 init_std=0.02,
				 eval_metrics='finetune',
				 **kwargs):
		super().__init__()
		self.init_std = init_std
		self.eval_metrics = eval_metrics
		self.cls_head = nn.Linear(in_channels, num_classes)
		
		self.init_weights(self.cls_head)

	def init_weights(self, module):
		if hasattr(module, 'weight') and module.weight is not None:
			if self.eval_metrics == 'finetune':
				trunc_normal_(module.weight, std=self.init_std)
			else:
				module.weight.data.normal_(mean=0.0, std=0.01)
		if hasattr(module, 'bias') and module.bias is not None:
			constant_init_(module.bias, constant_value=0)

	def forward(self, x):
		cls_score = self.cls_head(x)
		return cls_score


class PatchEmbed(nn.Module):
	"""Images to Patch Embedding.

	Args:
		img_size (int | tuple): Size of input image.
		patch_size (int): Size of one patch.
		tube_size (int): Size of temporal field of one 3D patch.
		in_channels (int): Channel num of input features. Defaults to 3.
		embed_dims (int): Dimensions of embedding. Defaults to 768.
		conv_type (str): Type for convolution layer. Defaults to 'Conv2d'.
	"""

	def __init__(self,
				 img_size,
				 patch_size,
				 tube_size=2,
				 in_channels=3,
				 embed_dims=768,
				 conv_type='Conv2d'):
		super().__init__()
		self.img_size = _pair(img_size)
		self.patch_size = _pair(patch_size)

		num_patches = \
			(self.img_size[1] // self.patch_size[1]) * \
			(self.img_size[0] // self.patch_size[0])
		assert (num_patches * self.patch_size[0] * self.patch_size[1] == 
			   self.img_size[0] * self.img_size[1],
			   'The image size H*W must be divisible by patch size')
		self.num_patches = num_patches

		# Use conv layer to embed
		if conv_type == 'Conv2d':
			self.projection = nn.Conv2d(
				in_channels,
				embed_dims,
				kernel_size=patch_size,
				stride=patch_size)
		elif conv_type == 'Conv3d':
			self.projection = nn.Conv3d(
				in_channels,
				embed_dims,
				kernel_size=(tube_size,patch_size,patch_size),
				stride=(tube_size,patch_size,patch_size))
		else:
			raise TypeError(f'Unsupported conv layer type {conv_type}')
			
		self.init_weights(self.projection)

	def init_weights(self, module):
		if hasattr(module, 'weight') and module.weight is not None:
			kaiming_init_(module.weight, mode='fan_in', nonlinearity='relu')
		if hasattr(module, 'bias') and module.bias is not None:
			constant_init_(module.bias, constant_value=0)

	def forward(self, x):
		layer_type = type(self.projection)
		if layer_type == nn.Conv3d:
			x = rearrange(x, 'b t c h w -> b c t h w')
			x = self.projection(x)
			x = rearrange(x, 'b c t h w -> (b t) (h w) c')
		elif layer_type == nn.Conv2d:
			x = rearrange(x, 'b t c h w -> (b t) c h w')
			x = self.projection(x)
			x = rearrange(x, 'b c h w -> b (h w) c')
		else:
			raise TypeError(f'Unsupported conv layer type {layer_type}')
		
		return x

class Attention(nn.Module):
	def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = qk_scale or head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x, attn

class DividedTemporalAttentionWithPreNorm(nn.Module):
	"""Temporal Attention in Divided Space Time Attention. 
		A warp for torch.nn.MultiheadAttention.

	Args:
		embed_dims (int): Dimensions of embedding.
		num_heads (int): Number of parallel attention heads in
			TransformerCoder.
		num_frames (int): Number of frames in the video.
		use_cls_token (bool): Whether to perform MSA on cls_token.
		attn_drop (float): A Dropout layer on attn_output_weights. Defaults to
			0..
		proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
			Defaults to 0..
		layer_drop (dict): The layer_drop used when adding the shortcut.
			Defaults to `dict(type=DropPath, dropout_p=0.1)`.
		norm_layer (class): Class name for normalization layer. Defaults to
			nn.LayerNorm.
	"""

	def __init__(self,
				 embed_dims,
				 num_heads,
				 num_frames,
				 use_cls_token,
				 attn_drop=0.,
				 proj_drop=0.,
				 layer_drop=dict(type=DropPath, dropout_p=0.1),
				 norm_layer=nn.LayerNorm,
				 **kwargs):
		super().__init__()
		self.embed_dims = embed_dims
		self.num_heads = num_heads
		self.num_frames = num_frames
		self.use_cls_token = use_cls_token

		self.norm = norm_layer(embed_dims)        
		#self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
		#								  **kwargs)
		self.attn = Attention(embed_dims, num_heads, qkv_bias=True, attn_drop=attn_drop) # batch first
										  
		self.proj_drop = nn.Dropout(proj_drop)
		dropout_p = layer_drop.pop('dropout_p')
		layer_drop= layer_drop.pop('type')
		self.layer_drop = layer_drop(dropout_p) if layer_drop else nn.Identity()
		if not use_cls_token:
			self.temporal_fc = nn.Linear(self.embed_dims, self.embed_dims)
			self.init_weights(self.temporal_fc)

	def init_weights(self, module):
		if hasattr(module, 'weight') and module.weight is not None:
			constant_init_(module.weight, constant_value=0)
		if hasattr(module, 'bias') and module.bias is not None:
			constant_init_(module.bias, constant_value=0)

	def forward(self, query, key=None, value=None, residual=None, return_attention=False, **kwargs):
		assert residual is None, (
			'Always adding the shortcut in the forward function')
		
		cls_token = query[:, 0, :].unsqueeze(1)
		if self.use_cls_token:
			residual = query
			query = query[:, 1:, :]
		else:
			query = query[:, 1:, :]
			residual = query

		b, n, d = query.size()
		p, t = n // self.num_frames, self.num_frames
		
		# Pre-Process
		query = rearrange(query, 'b (p t) d -> (b p) t d', p=p, t=t)
		if self.use_cls_token:
			cls_token = repeat(cls_token, 'b n d -> b (p n) d', p=p)
			cls_token = rearrange(cls_token, 'b p d -> (b p) 1 d')
			query = torch.cat((cls_token, query), 1)
		
		# Forward MSA
		query = self.norm(query)
		#query = rearrange(query, 'b n d -> n b d')
		#attn_out = self.attn(query, query, query)[0]
		#attn_out = rearrange(attn_out, 'n b d -> b n d')
		attn_out, attn_weights = self.attn(query)
		if return_attention:
			return attn_weights
		
		attn_out = self.layer_drop(self.proj_drop(attn_out.contiguous()))
		if not self.use_cls_token:
			attn_out = self.temporal_fc(attn_out)
		
		# Post-Process
		if self.use_cls_token:
			cls_token, attn_out = attn_out[:, 0, :], attn_out[:, 1:, :]
			cls_token = rearrange(cls_token, '(b p) d -> b p d', b=b)
			cls_token = reduce(cls_token, 'b p d -> b 1 d', 'mean')
			
			attn_out = rearrange(attn_out, '(b p) t d -> b (p t) d', p=p, t=t)
			attn_out = torch.cat((cls_token, attn_out), 1)
			new_query = residual + attn_out
		else:
			attn_out = rearrange(attn_out, '(b p) t d -> b (p t) d', p=p, t=t)
			new_query = residual + attn_out
			new_query = torch.cat((cls_token, new_query), 1)
		return new_query


class DividedSpatialAttentionWithPreNorm(nn.Module):
	"""Spatial Attention in Divided Space Time Attention.
		A warp for torch.nn.MultiheadAttention.
		
	Args:
		embed_dims (int): Dimensions of embedding.
		num_heads (int): Number of parallel attention heads in
			TransformerCoder.
		num_frames (int): Number of frames in the video.
		use_cls_token (bool): Whether to perform MSA on cls_token.
		attn_drop (float): A Dropout layer on attn_output_weights. Defaults to
			0..
		proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
			Defaults to 0..
		layer_drop (dict): The layer_drop used when adding the shortcut.
			Defaults to `dict(type=DropPath, dropout_p=0.1)`.
		norm_layer (class): Class name for normalization layer. Defaults to
			nn.LayerNorm.
	"""

	def __init__(self,
				 embed_dims,
				 num_heads,
				 num_frames,
				 use_cls_token,
				 attn_drop=0.,
				 proj_drop=0.,
				 layer_drop=dict(type=DropPath, dropout_p=0.1),
				 norm_layer=nn.LayerNorm,
				 **kwargs):
		super().__init__()
		self.embed_dims = embed_dims
		self.num_heads = num_heads
		self.num_frames = num_frames
		self.use_cls_token = use_cls_token
		
		self.norm = norm_layer(embed_dims)
		#self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
		#								  **kwargs)
		self.attn = Attention(embed_dims, num_heads, qkv_bias=True, attn_drop=attn_drop) # batch first
										  
		self.proj_drop = nn.Dropout(proj_drop)
		dropout_p = layer_drop.pop('dropout_p')
		layer_drop= layer_drop.pop('type')
		self.layer_drop = layer_drop(dropout_p) if layer_drop else nn.Identity()

		self.init_weights()

	def init_weights(self):
		pass

	def forward(self, query, key=None, value=None, residual=None, return_attention=False, **kwargs):
		assert residual is None, (
			'Always adding the shortcut in the forward function')
		
		cls_token = query[:, 0, :].unsqueeze(1)
		if self.use_cls_token:
			residual = query
			query = query[:, 1:, :]
		else:
			query = query[:, 1:, :]
			residual = query

		b, n, d = query.size()
		p, t = n // self.num_frames, self.num_frames
		
		# Pre-Process
		query = rearrange(query, 'b (p t) d -> (b t) p d', p=p, t=t)
		if self.use_cls_token:
			cls_token = repeat(cls_token, 'b n d -> b (t n) d', t=t)
			cls_token = rearrange(cls_token, 'b t d -> (b t) 1 d')
			query = torch.cat((cls_token, query), 1)
		
		# Forward MSA
		query = self.norm(query)
		#query = rearrange(query, 'b n d -> n b d')
		#attn_out = self.attn(query, query, query)[0]
		#attn_out = rearrange(attn_out, 'n b d -> b n d')
		attn_out, attn_weights = self.attn(query)
		if return_attention:
			return attn_weights

		attn_out = self.layer_drop(self.proj_drop(attn_out.contiguous()))
		
		# Post-Process
		if self.use_cls_token:
			cls_token, attn_out = attn_out[:, 0, :], attn_out[:, 1:, :]
			cls_token = rearrange(cls_token, '(b t) d -> b t d', b=b)
			cls_token = reduce(cls_token, 'b t d -> b 1 d', 'mean')
			
			attn_out = rearrange(attn_out, '(b t) p d -> b (p t) d', p=p, t=t)
			attn_out = torch.cat((cls_token, attn_out), 1)
			new_query = residual + attn_out
		else:
			attn_out = rearrange(attn_out, '(b t) p d -> b (p t) d', p=p, t=t)
			new_query = residual + attn_out
			new_query = torch.cat((cls_token, new_query), 1)
		return new_query


class MultiheadAttentionWithPreNorm(nn.Module):
	"""Implements MultiheadAttention with residual connection.
	
	Args:
		embed_dims (int): The embedding dimension.
		num_heads (int): Parallel attention heads.
		attn_drop (float): A Dropout layer on attn_output_weights.
			Default: 0.0.
		proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
			Default: 0.0.
		norm_layer (class): Class name for normalization layer. Defaults to
			nn.LayerNorm.
		layer_drop (obj:`ConfigDict`): The layer_drop used
			when adding the shortcut.
		batch_first (bool): When it is True,  Key, Query and Value are shape of
			(batch, n, embed_dim), otherwise (n, batch, embed_dim).
			 Default to False.
	"""

	def __init__(self,
				 embed_dims,
				 num_heads,
				 attn_drop=0.,
				 proj_drop=0.,
				 norm_layer=nn.LayerNorm,
				 layer_drop=dict(type=DropPath, dropout_p=0.),
				 batch_first=False,
				 **kwargs):
		super().__init__()
		self.embed_dims = embed_dims
		self.num_heads = num_heads
		#self.batch_first = batch_first
		
		self.norm = norm_layer(embed_dims)
		#self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
		#								  **kwargs)
		self.attn = Attention(embed_dims, num_heads, qkv_bias=True, attn_drop=attn_drop) # batch first

		self.proj_drop = nn.Dropout(proj_drop)
		dropout_p = layer_drop.pop('dropout_p')
		layer_drop= layer_drop.pop('type')
		self.layer_drop = layer_drop(dropout_p) if layer_drop else nn.Identity()

	def forward(self,
				query,
				key=None,
				value=None,
				residual=None,
				attn_mask=None,
				key_padding_mask=None,
				return_attention=False,
				**kwargs):
		residual = query
		
		query = self.norm(query)
		#if self.batch_first:
		#	query = query.transpose(0, 1)
		#attn_out = self.attn(
		#	query=query,
		#	key=query,
		#	value=query,
		#	attn_mask=attn_mask,
		#	key_padding_mask=key_padding_mask)[0]
		#attn_out = self.attn(query, query, query)[0]
		#if self.batch_first:
		#	attn_out = attn_out.transpose(0, 1)
		attn_out, attn_weights = self.attn(query)
		if return_attention:
			return attn_weights

		new_query = residual + self.layer_drop(self.proj_drop(attn_out))
		return new_query


class FFNWithPreNorm(nn.Module):
	"""Implements feed-forward networks (FFNs) with residual connection.
	
	Args:
		embed_dims (int): The feature dimension. Same as
			`MultiheadAttention`. Defaults: 256.
		hidden_channels (int): The hidden dimension of FFNs.
			Defaults: 1024.
		num_layers (int, optional): The number of fully-connected layers in
			FFNs. Default: 2.
		act_layer (dict, optional): The activation layer for FFNs.
			Default: nn.GELU
		norm_layer (class): Class name for normalization layer. Defaults to
			nn.LayerNorm.
		dropout_p (float, optional): Probability of an element to be
			zeroed in FFN. Default 0.0.
		layer_drop (obj:`ConfigDict`): The layer_drop used
			when adding the shortcut.
	"""
	
	def __init__(self,
				 embed_dims=256,
				 hidden_channels=1024,
				 num_layers=2,
				 act_layer=nn.GELU,
				 norm_layer=nn.LayerNorm,
				 dropout_p=0.,
				 layer_drop=None,
				 **kwargs):
		super().__init__()
		assert num_layers >= 2, 'num_layers should be no less ' \
			f'than 2. got {num_layers}.'
		self.embed_dims = embed_dims
		self.hidden_channels = hidden_channels
		self.num_layers = num_layers
		
		self.norm = norm_layer(embed_dims)
		layers = []
		in_channels = embed_dims
		for _ in range(num_layers - 1):
			layers.append(
				nn.Sequential(
					nn.Linear(in_channels, hidden_channels),
					act_layer(),
					nn.Dropout(dropout_p)))
			in_channels = hidden_channels
		layers.append(nn.Linear(hidden_channels, embed_dims))
		layers.append(nn.Dropout(dropout_p))
		self.layers = nn.ModuleList(layers)
		
		if layer_drop:
			dropout_p = layer_drop.pop('dropout_p')
			layer_drop= layer_drop.pop('type')
			self.layer_drop = layer_drop(dropout_p)  
		else:
			self.layer_drop = nn.Identity()

	def forward(self, x):
		residual = x
		
		x = self.norm(x)
		for layer in self.layers:
			x = layer(x)
			
		return residual + self.layer_drop(x)


class TransformerContainer(nn.Module):

	def __init__(self, 
				 num_transformer_layers,
				 embed_dims,
				 num_heads,
				 num_frames,
				 hidden_channels,
				 operator_order,
				 drop_path_rate=0.1,
				 norm_layer=nn.LayerNorm,
				 act_layer=nn.GELU,
				 num_layers=2):
		super().__init__()
		self.layers = nn.ModuleList([])
		self.num_transformer_layers = num_transformer_layers
		
		dpr = np.linspace(0, drop_path_rate, num_transformer_layers)
		for i in range(num_transformer_layers):	
			self.layers.append(
				BasicTransformerBlock(
					embed_dims=embed_dims,
					num_heads=num_heads,
					num_frames=num_frames,
					hidden_channels=hidden_channels,
					operator_order=operator_order,
					norm_layer=norm_layer,
					act_layer=act_layer,
					num_layers=num_layers,
					dpr=dpr[i]))
		
	def forward(self, x, return_attention=False):
		layer_idx = 0
		for layer in self.layers:
			if layer_idx >= self.num_transformer_layers-1 and return_attention:
				x = layer(x, return_attention=True)
			else:
				x = layer(x)
			layer_idx += 1
		return x


class BasicTransformerBlock(nn.Module):

	def __init__(self, 
				 embed_dims,
				 num_heads,
				 num_frames,
				 hidden_channels,
				 operator_order,
				 norm_layer=nn.LayerNorm,
				 act_layer=nn.GELU,
				 num_layers=2,
				 dpr=0,
				 ):

		super().__init__()
		self.attentions = nn.ModuleList([])
		self.ffns = nn.ModuleList([])
		
		for i, operator in enumerate(operator_order):
			if operator == 'self_attn':
				self.attentions.append(
					MultiheadAttentionWithPreNorm(
						embed_dims=embed_dims,
						num_heads=num_heads,
						batch_first=True,
						norm_layer=nn.LayerNorm,
						layer_drop=dict(type=DropPath, dropout_p=dpr)))
			elif operator == 'time_attn':
				self.attentions.append(
					DividedTemporalAttentionWithPreNorm(
						embed_dims=embed_dims,
						num_heads=num_heads,
						num_frames=num_frames,
						norm_layer=norm_layer,
						use_cls_token=(i==len(operator_order)-2),
						layer_drop=dict(type=DropPath, dropout_p=dpr)))
			elif operator == 'space_attn':
				self.attentions.append(
					DividedSpatialAttentionWithPreNorm(
						embed_dims=embed_dims,
						num_heads=num_heads,
						num_frames=num_frames,
						norm_layer=norm_layer,
						use_cls_token=(i==len(operator_order)-2),
						layer_drop=dict(type=DropPath, dropout_p=dpr)))
			elif operator == 'ffn':
				self.ffns.append(
					FFNWithPreNorm(
						embed_dims=embed_dims,
						hidden_channels=hidden_channels,
						num_layers=num_layers,
						act_layer=act_layer,
						norm_layer=norm_layer,
						layer_drop=dict(type=DropPath, dropout_p=dpr)))
			else:
				raise TypeError(f'Unsupported operator type {operator}')
		
	def forward(self, x, return_attention=False):
		attention_idx = 0
		for layer in self.attentions:
			if attention_idx >= len(self.attentions)-1 and return_attention:
				x = layer(x, return_attention=True)
				return x
			else:
				x = layer(x)
			attention_idx += 1
		for layer in self.ffns:
			x = layer(x)
		return x
