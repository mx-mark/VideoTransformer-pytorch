from einops import rearrange, reduce, repeat
import torch
import torch.nn as nn

from transformer import PatchEmbed, TransformerContainer, get_sine_cosine_pos_emb
from weight_init import (trunc_normal_, init_from_vit_pretrain_, 
	init_from_mae_pretrain_, init_from_k600_pretrain_)

from typing import Callable, Tuple
from pytorchvideo.layers.utils import set_attributes
from pytorchvideo.models.vision_transformers import create_multiscale_vision_transformers

def round_width(width, multiplier, min_width=1, divisor=1):
	if not multiplier:
		return width
	width *= multiplier
	min_width = min_width or divisor

	width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
	if width_out < 0.9 * width:
		width_out += divisor
	return int(width_out)

class TimeSformer(nn.Module):
	"""TimeSformer. A PyTorch impl of `Is Space-Time Attention All You Need for
	Video Understanding? <https://arxiv.org/abs/2102.05095>`_

	Args:
		num_frames (int): Number of frames in the video.
		img_size (int | tuple): Size of input image.
		patch_size (int): Size of one patch.
		pretrained (str | None): Name of pretrained model. Default: None.
		embed_dims (int): Dimensions of embedding. Defaults to 768.
		num_heads (int): Number of parallel attention heads in
			TransformerCoder. Defaults to 12.
		num_transformer_layers (int): Number of transformer layers. Defaults to
			12.
		in_channels (int): Channel num of input features. Defaults to 3.
		dropout_p (float): Probability of dropout layer. Defaults to 0.
		conv_type (str): Type of the convolution in PatchEmbed layer. Defaults to Conv2d.
		attention_type (str): Type of attentions in TransformerCoder. Choices
			are 'divided_space_time', 'space_only' and 'joint_space_time'.
			Defaults to 'divided_space_time'.
		norm_layer (dict): Config for norm layers. Defaults to nn.LayerNorm.
		copy_strategy (str): Copy or Initial to zero towards the new additional layer.
		use_learnable_pos_emb (bool): Whether to use learnable position embeddings.
		return_cls_token (bool): Whether to use cls_token to predict class label.
	"""
	supported_attention_types = [
		'divided_space_time', 'space_only', 'joint_space_time'
	]

	def __init__(self,
				 num_frames,
				 img_size,
				 patch_size,
				 pretrained=None,
				 embed_dims=768,
				 num_heads=12,
				 num_transformer_layers=12,
				 in_channels=3,
				 conv_type='Conv2d',
				 dropout_p=0.,
				 attention_type='divided_space_time',
				 norm_layer=nn.LayerNorm,
				 copy_strategy='repeat',
				 use_learnable_pos_emb=True,
				 return_cls_token=True,
				 **kwargs):
		super().__init__()
		assert attention_type in self.supported_attention_types, (
			f'Unsupported Attention Type {attention_type}!')

		self.num_frames = num_frames
		self.pretrained = pretrained
		self.embed_dims = embed_dims
		self.num_transformer_layers = num_transformer_layers
		self.attention_type = attention_type
		self.copy_strategy = copy_strategy
		self.conv_type = conv_type
		self.use_learnable_pos_emb = use_learnable_pos_emb
		self.return_cls_token = return_cls_token

		#tokenize & position embedding
		self.patch_embed = PatchEmbed(
			img_size=img_size,
			patch_size=patch_size,
			in_channels=in_channels,
			embed_dims=embed_dims,
			conv_type=conv_type)
		num_patches = self.patch_embed.num_patches
		
		if self.attention_type == 'divided_space_time':
			# Divided Space Time Attention
			operator_order = ['time_attn','space_attn','ffn']
			container = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=operator_order)

			transformer_layers = container
		else:
			# Sapce Only & Joint Space Time Attention
			operator_order = ['self_attn','ffn']
			container = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=operator_order)

			transformer_layers = container

		self.transformer_layers = transformer_layers
		self.norm = norm_layer(embed_dims, eps=1e-6)
		
		self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dims))
		# whether to add one cls_token in temporal pos_emb
		self.use_cls_token_temporal = operator_order[-2] == 'time_attn'
		if self.use_cls_token_temporal:
			num_frames = num_frames + 1
		else:
			num_patches = num_patches + 1

		# spatial pos_emb
		if use_learnable_pos_emb:
			self.pos_embed = nn.Parameter(torch.zeros(1,num_patches,embed_dims))
		else:
			self.pos_embed = get_sine_cosine_pos_emb(num_patches,embed_dims)
		self.drop_after_pos = nn.Dropout(p=dropout_p)
		
		# temporal pos_emb
		if self.attention_type != 'space_only':	
			if use_learnable_pos_emb:
				self.time_embed = nn.Parameter(torch.zeros(1,num_frames,embed_dims))
			else:
				self.time_embed = get_sine_cosine_pos_emb(num_frames,embed_dims)
			self.drop_after_time = nn.Dropout(p=dropout_p)

		self.init_weights()

	def init_weights(self):
		if self.use_learnable_pos_emb:
			#trunc_normal_(self.pos_embed, std=.02)
			nn.init.trunc_normal_(self.pos_embed, std=.02)
			if self.attention_type != 'space_only':
				nn.init.trunc_normal_(self.time_embed, std=.02)
		trunc_normal_(self.cls_token, std=.02)
		
		if self.pretrained is not None:
			if 'mae' in self.pretrained:
				init_from_mae_pretrain_(self,
										self.pretrained,
										self.conv_type,
										self.attention_type,
										self.copy_strategy)
			elif 'vit' in self.pretrained:
				init_from_vit_pretrain_(self,
										self.pretrained,
										self.conv_type,
										self.attention_type,
										self.copy_strategy)
			elif 'timesformer' in self.pretrained:
				init_from_k600_pretrain_(self,
										 self.pretrained,
										 'transformer')
			else:
				raise TypeError(f'not support the pretrained weight {self.pretrained}')

	def forward(self, x):
		#Tokenize
		b = x.shape[0]
		x = self.patch_embed(x)
		
		# Add Position Embedding
		cls_tokens = repeat(self.cls_token, 'b ... -> (repeat b) ...', repeat=x.shape[0])
		if self.use_cls_token_temporal:
			if self.use_learnable_pos_emb:
				x = x + self.pos_embed
			else:
				x = x + self.pos_embed.type_as(x).detach()
			x = torch.cat((cls_tokens, x), dim=1)
		else:
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.pos_embed
			else:
				x = x + self.pos_embed.type_as(x).detach()
		x = self.drop_after_pos(x)

		# Add Time Embedding
		if self.attention_type != 'space_only':
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			if self.use_cls_token_temporal:
				x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
				cls_tokens = repeat(cls_tokens, 
									'b ... -> (repeat b) ...',
									repeat=x.shape[0]//b)
				x = torch.cat((cls_tokens, x), dim=1)
				if self.use_learnable_pos_emb:
					x = x + self.time_embed
				else:
					x = x + self.time_embed.type_as(x).detach()
				cls_tokens = x[:b, 0, :].unsqueeze(1)
				x = rearrange(x[:, 1:, :], '(b p) t d -> b (p t) d', b=b)
				x = torch.cat((cls_tokens, x), dim=1)
			else:
				x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
				if self.use_learnable_pos_emb:
					x = x + self.time_embed
				else:
					x = x + self.time_embed.type_as(x).detach()
				x = rearrange(x, '(b p) t d -> b (p t) d', b=b)
				x = torch.cat((cls_tokens, x), dim=1)
			x = self.drop_after_time(x)
		
		# Video transformer forward
		x = self.transformer_layers(x)

		if self.attention_type == 'space_only':
			x = rearrange(x, '(b t) p d -> b t p d', b=b)
			x = reduce(x, 'b t p d -> b p d', 'mean')

		x = self.norm(x)
		# Return Class Token
		if self.return_cls_token:
			return x[:, 0]
		else:
			return x[:, 1:].mean(1)


class ViViT(nn.Module):
	"""ViViT. A PyTorch impl of `ViViT: A Video Vision Transformer`
		<https://arxiv.org/abs/2103.15691>

	Args:
		num_frames (int): Number of frames in the video.
		img_size (int | tuple): Size of input image.
		patch_size (int): Size of one patch.
		pretrained (str | None): Name of pretrained model. Default: None.
		embed_dims (int): Dimensions of embedding. Defaults to 768.
		num_heads (int): Number of parallel attention heads. Defaults to 12.
		num_transformer_layers (int): Number of transformer layers. Defaults to 12.
		in_channels (int): Channel num of input features. Defaults to 3.
		dropout_p (float): Probability of dropout layer. Defaults to 0..
		tube_size (int): Dimension of the kernel size in Conv3d. Defaults to 2.
		conv_type (str): Type of the convolution in PatchEmbed layer. Defaults to Conv3d.
		attention_type (str): Type of attentions in TransformerCoder. Choices
			are 'divided_space_time', 'fact_encoder' and 'joint_space_time'.
			Defaults to 'fact_encoder'.
		norm_layer (dict): Config for norm layers. Defaults to nn.LayerNorm.
		copy_strategy (str): Copy or Initial to zero towards the new additional layer.
		extend_strategy (str): How to initialize the weights of Conv3d from pre-trained Conv2d.
		use_learnable_pos_emb (bool): Whether to use learnable position embeddings.
		return_cls_token (bool): Whether to use cls_token to predict class label.
	"""
	supported_attention_types = [
		'fact_encoder', 'joint_space_time', 'divided_space_time'
	]

	def __init__(self,
				 num_frames,
				 img_size,
				 patch_size,
				 pretrained=None,
				 embed_dims=768,
				 num_heads=12,
				 num_transformer_layers=12,
				 in_channels=3,
				 dropout_p=0.,
				 tube_size=2,
				 conv_type='Conv3d',
				 attention_type='fact_encoder',
				 norm_layer=nn.LayerNorm,
				 copy_strategy='repeat',
				 extend_strategy='temporal_avg',
				 use_learnable_pos_emb=True,
				 return_cls_token=True,
				 **kwargs):
		super().__init__()
		assert attention_type in self.supported_attention_types, (
			f'Unsupported Attention Type {attention_type}!')
		
		num_frames = num_frames//tube_size
		self.num_frames = num_frames
		self.pretrained = pretrained
		self.embed_dims = embed_dims
		self.num_transformer_layers = num_transformer_layers
		self.attention_type = attention_type
		self.conv_type = conv_type
		self.copy_strategy = copy_strategy
		self.extend_strategy = extend_strategy
		self.tube_size = tube_size
		self.num_time_transformer_layers = 0
		self.use_learnable_pos_emb = use_learnable_pos_emb
		self.return_cls_token = return_cls_token

		#tokenize & position embedding
		self.patch_embed = PatchEmbed(
			img_size=img_size,
			patch_size=patch_size,
			in_channels=in_channels,
			embed_dims=embed_dims,
			tube_size=tube_size,
			conv_type=conv_type)
		num_patches = self.patch_embed.num_patches
		
		if self.attention_type == 'divided_space_time':
			# Divided Space Time Attention - Model 3
			operator_order = ['time_attn','space_attn','ffn']
			container = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=operator_order)

			transformer_layers = container
		elif self.attention_type == 'joint_space_time':
			# Joint Space Time Attention - Model 1
			operator_order = ['self_attn','ffn']
			container = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=operator_order)
			
			transformer_layers = container
		else:
			# Divided Space Time Transformer Encoder - Model 2
			transformer_layers = nn.ModuleList([])
			self.num_time_transformer_layers = 4
			
			spatial_transformer = TransformerContainer(
				num_transformer_layers=num_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=['self_attn','ffn'])
			
			temporal_transformer = TransformerContainer(
				num_transformer_layers=self.num_time_transformer_layers,
				embed_dims=embed_dims,
				num_heads=num_heads,
				num_frames=num_frames,
				norm_layer=norm_layer,
				hidden_channels=embed_dims*4,
				operator_order=['self_attn','ffn'])

			transformer_layers.append(spatial_transformer)
			transformer_layers.append(temporal_transformer)
 
		self.transformer_layers = transformer_layers
		self.norm = norm_layer(embed_dims, eps=1e-6)
		
		self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dims))
		# whether to add one cls_token in temporal pos_enb
		if attention_type == 'fact_encoder':
			num_frames = num_frames + 1
			num_patches = num_patches + 1
			self.use_cls_token_temporal = False
		else:
			self.use_cls_token_temporal = operator_order[-2] == 'time_attn'
			if self.use_cls_token_temporal:
				num_frames = num_frames + 1
			else:
				num_patches = num_patches + 1

		if use_learnable_pos_emb:
			self.pos_embed = nn.Parameter(torch.zeros(1,num_patches,embed_dims))
			self.time_embed = nn.Parameter(torch.zeros(1,num_frames,embed_dims))
		else:
			self.pos_embed = get_sine_cosine_pos_emb(num_patches,embed_dims)
			self.time_embed = get_sine_cosine_pos_emb(num_frames,embed_dims)
		self.drop_after_pos = nn.Dropout(p=dropout_p)
		self.drop_after_time = nn.Dropout(p=dropout_p)

		self.init_weights()

	def init_weights(self):
		if self.use_learnable_pos_emb:
			#trunc_normal_(self.pos_embed, std=.02)
			#trunc_normal_(self.time_embed, std=.02)
			nn.init.trunc_normal_(self.pos_embed, std=.02)
			nn.init.trunc_normal_(self.time_embed, std=.02)
		trunc_normal_(self.cls_token, std=.02)
		
		if self.pretrained is not None:
			if 'mae' in self.pretrained:
				init_from_mae_pretrain_(self,
										self.pretrained,
										self.conv_type,
										self.attention_type,
										self.copy_strategy,
										self.extend_strategy, 
										self.tube_size, 
										self.num_time_transformer_layers)
			elif 'vit' in self.pretrained:
				init_from_vit_pretrain_(self,
										self.pretrained,
										self.conv_type,
										self.attention_type,
										self.copy_strategy,
										self.extend_strategy, 
										self.tube_size, 
										self.num_time_transformer_layers)
			else:
				raise TypeError(f'not support the pretrained weight {self.pretrained}')

	def forward(self, x):
		#Tokenize
		b = x.shape[0]
		x = self.patch_embed(x)
		
		# Add Position Embedding
		cls_tokens = repeat(self.cls_token, 'b ... -> (repeat b) ...', repeat=x.shape[0])
		if self.use_cls_token_temporal:
			if self.use_learnable_pos_emb:
				x = x + self.pos_embed
			else:
				x = x + self.pos_embed.type_as(x).detach()
			x = torch.cat((cls_tokens, x), dim=1)
		else:
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.pos_embed
			else:
				x = x + self.pos_embed.type_as(x).detach()
		x = self.drop_after_pos(x)
		
		# Add Time Embedding
		if self.attention_type != 'fact_encoder':
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			if self.use_cls_token_temporal:
				x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
				cls_tokens = repeat(cls_tokens,
									'b ... -> (repeat b) ...',
									repeat=x.shape[0]//b)
				x = torch.cat((cls_tokens, x), dim=1)
				if self.use_learnable_pos_emb:
					x = x + self.time_embed
				else:
					x = x + self.time_embed.type_as(x).detach()
				cls_tokens = x[:b, 0, :].unsqueeze(1)
				x = rearrange(x[:, 1:, :], '(b p) t d -> b (p t) d', b=b)
				x = torch.cat((cls_tokens, x), dim=1)
			else:
				x = rearrange(x[:, 1:, :], '(b t) p d -> (b p) t d', b=b)
				if self.use_learnable_pos_emb:
					x = x + self.time_embed
				else:
					x = x + self.time_embed.type_as(x).detach()
				x = rearrange(x, '(b p) t d -> b (p t) d', b=b)
				x = torch.cat((cls_tokens, x), dim=1)
			x = self.drop_after_time(x)
			
			x = self.transformer_layers(x)
		else:
			# fact encoder - CRNN style
			spatial_transformer, temporal_transformer, = *self.transformer_layers,
			x = spatial_transformer(x)
			
			# Add Time Embedding
			cls_tokens = x[:b, 0, :].unsqueeze(1)
			x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
			x = reduce(x, 'b t p d -> b t d', 'mean')
			x = torch.cat((cls_tokens, x), dim=1)
			if self.use_learnable_pos_emb:
				x = x + self.time_embed
			else:
				x = x + self.time_embed.type_as(x).detach()
			x = self.drop_after_time(x)
			
			x = temporal_transformer(x)

		x = self.norm(x)
		# Return Class Token
		if self.return_cls_token:
			return x[:, 0]
		else:
			return x[:, 1:].mean(1)

class PatchEmbeding(nn.Module):
	"""
	Transformer basic patch embedding module. Performs patchifying input, flatten and
	and transpose.
	The builder can be found in `create_patch_embed`.
	"""

	def __init__(
		self,
		*,
		patch_model: nn.Module = None,
	) -> None:
		super().__init__()
		set_attributes(self, locals())
		assert self.patch_model is not None

	def forward(self, x) -> torch.Tensor:
		x = self.patch_model(x)
		# B C (T) H W -> B (T)HW C
		return x.flatten(2).transpose(1, 2)

def create_conv_patch_embed(
	*,
	in_channels: int,
	out_channels: int,
	conv_kernel_size: Tuple[int] = (1, 16, 16),
	conv_stride: Tuple[int] = (1, 4, 4),
	conv_padding: Tuple[int] = (1, 7, 7),
	conv_bias: bool = True,
	conv: Callable = nn.Conv3d,
) -> nn.Module:
	"""
	Creates the transformer basic patch embedding. It performs Convolution, flatten and
	transpose.
	Args:
		in_channels (int): input channel size of the convolution.
		out_channels (int): output channel size of the convolution.
		conv_kernel_size (tuple): convolutional kernel size(s).
		conv_stride (tuple): convolutional stride size(s).
		conv_padding (tuple): convolutional padding size(s).
		conv_bias (bool): convolutional bias. If true, adds a learnable bias to the
			output.
		conv (callable): Callable used to build the convolution layer.
	Returns:
		(nn.Module): transformer patch embedding layer.
	"""
	conv_module = conv(
		in_channels=in_channels,
		out_channels=out_channels,
		kernel_size=conv_kernel_size,
		stride=conv_stride,
		padding=conv_padding,
		bias=conv_bias,
	)
	return PatchEmbeding(patch_model=conv_module)

class MaskFeat(nn.Module):
	"""
	Multiscale Vision Transformers
	Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
	https://arxiv.org/abs/2104.11227
	"""

	def __init__(self,
				 img_size=224,
				 num_frames=16,
				 input_channels=3,
				 patch_embed_dim=96,
				 conv_patch_embed_kernel=(3, 7, 7),
				 conv_patch_embed_stride=(2, 4, 4),
				 conv_patch_embed_padding=(1, 3, 3),
				 embed_dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
				 atten_head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
				 pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
				 pool_kv_stride_adaptive=[1, 8, 8],
				 pool_kvq_kernel=[3, 3, 3],
				 head=None,
				 feature_dim=10,
				 **kwargs):
		super().__init__()
		self.num_frames = num_frames
		self.img_size = img_size
		self.stride = conv_patch_embed_stride
		self.downsample_rate = 2 ** len(pool_q_stride_size)
		# Get mvit from pytorchvideo
		self.patch_embed = (
			create_conv_patch_embed(
				in_channels=input_channels,
				out_channels=patch_embed_dim,
				conv_kernel_size=conv_patch_embed_kernel,
				conv_stride=conv_patch_embed_stride,
				conv_padding=conv_patch_embed_padding,
				conv=nn.Conv3d,
			)
		)
		self.mvit = create_multiscale_vision_transformers(
			spatial_size=img_size, 
			temporal_size=num_frames,
			embed_dim_mul=embed_dim_mul,
			atten_head_mul=atten_head_mul,
			pool_q_stride_size=pool_q_stride_size,
			pool_kv_stride_adaptive=pool_kv_stride_adaptive,
			pool_kvq_kernel=pool_kvq_kernel,
			enable_patch_embed=False,
			head=head)
		in_features = self.mvit.norm_embed.normalized_shape[0]
		out_features = feature_dim # the dimension of the predict features
		self.decoder_pred = nn.Linear(in_features, feature_dim, bias=True)
		# mask token
		self.mask_token = nn.Parameter(torch.zeros(1, 1, patch_embed_dim))

		# init weights
		w = self.patch_embed.patch_model.weight.data
		nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
		nn.init.xavier_uniform_(self.decoder_pred.weight)
		nn.init.constant_(self.decoder_pred.bias, 0)
		nn.init.trunc_normal_(self.mask_token, std=.02)

	def forward(self, x, target_x, mask, cube_marker, visualize=False):
		x = self.patch_embed(x.transpose(1,2))
		# apply mask tokens
		B, L, C = x.shape
		mask_token = self.mask_token.expand(B, L, -1)
		dense_mask = repeat(mask, 'b t h w -> b t (h dh) (w dw)', dh=self.downsample_rate, dw=self.downsample_rate) # nearest-neighbor resize
		w = dense_mask.flatten(1).unsqueeze(-1).type_as(mask_token)
		x = x * (1 - w) + mask_token * w
		
		# forward network
		x = self.mvit(x)
		x = self.decoder_pred(x)
		x = x[:, 1:, :]
		
		# reshape to original x
		x = rearrange(x, 'b (t h w) (dt dc) -> b (t dt) h w dc', 
			dt=self.stride[0],
			t=self.num_frames//self.stride[0],
			h=self.img_size//(self.stride[1]*self.downsample_rate),
			w=self.img_size//(self.stride[2]*self.downsample_rate))
		
		# find the center frame of the mask cube
		mask = repeat(mask, 'b t h w -> b (t dt) h w', dt=self.stride[0])
		center_index = torch.zeros(self.num_frames).to(torch.bool)
		for marker in cube_marker: # [[start, span]]
			start_frame, span_frame = marker[0]
			center_index[start_frame*self.stride[0] + span_frame*self.stride[0]//2] = 1 # center index extends to 16
		mask[:, ~center_index] = 0
		
		# compute loss on mask regions in center frame
		loss = (x - target_x) ** 2
		loss = loss.mean(dim=-1)
		loss = (loss * mask).sum() / (mask.sum() + 1e-5)
		
		# visulize 
		if visualize:
			mask_preds = x[:, center_index]
			mask_preds = rearrange(mask_preds, 'b t h w (dh dw c o) -> b t (h dh) (w dw) c o', dh=2,dw=2,c=3,o=9) # need to unnormalize
			return x, loss, mask_preds, center_index
		else:
			return x, loss


def visualize_hog(orientation_histogram, save_path, s_row=224, s_col=224, orientations=9, c_row=8, c_col=8):
	n_cells_row = int(s_row // c_row)  # number of cells along row-axis
	n_cells_col = int(s_col // c_col)  # number of cells along col-axis
	radius = min(c_row, c_col) // 2 - 1
	orientations_arr = np.arange(orientations)
	# set dr_arr, dc_arr to correspond to midpoints of orientation bins
	orientation_bin_midpoints = (
		np.pi * (orientations_arr + .5) / orientations)
	dr_arr = radius * np.sin(orientation_bin_midpoints)
	dc_arr = radius * np.cos(orientation_bin_midpoints)
	hog_image = np.zeros((s_row, s_col), dtype=np.float64)
	for r in range(n_cells_row):
		for c in range(n_cells_col):
			for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
				centre = tuple([r * c_row + c_row // 2,
								c * c_col + c_col // 2])
				rr, cc = draw.line(int(centre[0] - dc),
								   int(centre[1] + dr),
								   int(centre[0] + dc),
								   int(centre[1] - dr))
				hog_image[rr, cc] += orientation_histogram[r, c, o]
	io.imsave(save_path, hog_image)


def _hog_normalize_block(block, eps=1e-5):
	return np.sqrt(np.sum(block ** 2) + eps ** 2)


def _hog_channel_gradient(channel):
	"""Compute unnormalized gradient image along `row` and `col` axes.

	Parameters
	----------
	channel : (M, N) ndarray
		Grayscale image or one of image channel.

	Returns
	-------
	g_row, g_col : channel gradient along `row` and `col` axes correspondingly.
	"""
	g_row = np.empty(channel.shape, dtype=channel.dtype)
	g_row[0, :] = 0
	g_row[-1, :] = 0
	g_row[1:-1, :] = channel[2:, :] - channel[:-2, :]
	g_col = np.empty(channel.shape, dtype=channel.dtype)
	g_col[:, 0] = 0
	g_col[:, -1] = 0
	g_col[:, 1:-1] = channel[:, 2:] - channel[:, :-2]

	return g_row, g_col


from skimage._shared import utils
@utils.channel_as_last_axis(multichannel_output=False)
@utils.deprecate_multichannel_kwarg(multichannel_position=8)
def get_hog_norm(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
		transform_sqrt=False, multichannel=None, *, channel_axis=None):
	#Extract Histogram of Oriented Gradients (HOG) for a given image.
	image = np.atleast_2d(image)
	float_dtype = utils._supported_float_type(image.dtype)
	image = image.astype(float_dtype, copy=False)

	multichannel = channel_axis is not None
	ndim_spatial = image.ndim - 1 if multichannel else image.ndim
	if ndim_spatial != 2:
		raise ValueError('Only images with two spatial dimensions are '
						 'supported. If using with color/multichannel '
						 'images, specify `channel_axis`.')

	"""
	The first stage applies an optional global image normalization
	equalisation that is designed to reduce the influence of illumination
	effects. In practice we use gamma (power law) compression, either
	computing the square root or the log of each color channel.
	Image texture strength is typically proportional to the local surface
	illumination so this compression helps to reduce the effects of local
	shadowing and illumination variations.
	"""

	if transform_sqrt:
		image = np.sqrt(image)

	"""
	The second stage computes first order image gradients. These capture
	contour, silhouette and some texture information, while providing
	further resistance to illumination variations. The locally dominant
	color channel is used, which provides color invariance to a large
	extent. Variant methods may also include second order image derivatives,
	which act as primitive bar detectors - a useful feature for capturing,
	e.g. bar like structures in bicycles and limbs in humans.
	"""

	g_row, g_col = _hog_channel_gradient(image)

	"""
	The third stage aims to produce an encoding that is sensitive to
	local image content while remaining resistant to small changes in
	pose or appearance. The adopted method pools gradient orientation
	information locally in the same way as the SIFT [Lowe 2004]
	feature. The image window is divided into small spatial regions,
	called "cells". For each cell we accumulate a local 1-D histogram
	of gradient or edge orientations over all the pixels in the
	cell. This combined cell-level 1-D histogram forms the basic
	"orientation histogram" representation. Each orientation histogram
	divides the gradient angle range into a fixed number of
	predetermined bins. The gradient magnitudes of the pixels in the
	cell are used to vote into the orientation histogram.
	"""

	s_row, s_col = image.shape[:2]
	c_row, c_col = pixels_per_cell
	b_row, b_col = cells_per_block

	n_cells_row = int(s_row // c_row)  # number of cells along row-axis
	n_cells_col = int(s_col // c_col)  # number of cells along col-axis

	# compute orientations integral images
	orientation_histogram = np.zeros((n_cells_row, n_cells_col, orientations),
									 dtype=float)
	g_row = g_row.astype(float, copy=False)
	g_col = g_col.astype(float, copy=False)

	_hoghistogram.hog_histograms(g_col, g_row, c_col, c_row, s_col, s_row,
								 n_cells_col, n_cells_row,
								 orientations, orientation_histogram)

	"""
	The fourth stage computes normalization, which takes local groups of
	cells and contrast normalizes their overall responses before passing
	to next stage. Normalization introduces better invariance to illumination,
	shadowing, and edge contrast. It is performed by accumulating a measure
	of local histogram "energy" over local groups of cells that we call
	"blocks". The result is used to normalize each cell in the block.
	Typically each individual cell is shared between several blocks, but
	its normalizations are block dependent and thus different. The cell
	thus appears several times in the final output vector with different
	normalizations. This may seem redundant but it improves the performance.
	We refer to the normalized block descriptors as Histogram of Oriented
	Gradient (HOG) descriptors.
	"""

	n_blocks_row = (n_cells_row - b_row) + 1
	n_blocks_col = (n_cells_col - b_col) + 1
	normalized_blocks = np.zeros(
		(n_blocks_row, n_blocks_col, 1),
		dtype=float_dtype
	)

	for r in range(n_blocks_row):
		for c in range(n_blocks_col):
			block = orientation_histogram[r:r + b_row, c:c + b_col, :]
			normalized_blocks[r, c, :] = \
				_hog_normalize_block(block)

	return normalized_blocks


if __name__ == '__main__':
	# Unit test for model runnable experiment and Hog prediction
	import numpy as np
	from mask_generator import CubeMaskGenerator
	import data_transform as T
	from dataset import DecordInit, extract_hog_features
	from skimage import io, draw
	from skimage.feature import _hoghistogram
	from torchvision.transforms import ToPILImage
	device = torch.device('cuda:0')
	
	'''
	model = TimeSformer(num_frames=4,
						img_size=224,
						patch_size=16,
						pretrained='./pretrain_model/pretrain_mae_vit_base_mask_0.75_400e.pth',
						attention_type='divided_space_time',
						use_learnable_pos_emb=True,
						return_cls_token=True)
	'''
	'''
	model = ViViT(num_frames=4, 
				  img_size=224,
				  patch_size=16,
				  pretrained='./pretrain_model/pretrain_mae_vit_base_mask_0.75_400e.pth',
				  attention_type='divided_space_time',
				  use_learnable_pos_emb=False,
				  return_cls_token=False)
	'''
	# To be reproducable
	'''
	import random 
	SEED = 0
	torch.random.manual_seed(SEED)
	np.random.seed(SEED)
	random.seed(SEED)
	'''
	# 1. laod pretrained model
	from weight_init import replace_state_dict
	model_name = 'maskfeat'
	model = MaskFeat(pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2]], feature_dim=2*2*2*3*9)
	state_dict = torch.load(f'./{model_name}_model.pth')['state_dict']
	
	replace_state_dict(state_dict)
	missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
	print(missing_keys, unexpected_keys)
	model.eval()
	model = model.to(device)
	
	# 2. prepare data
	#input = torch.rand(2,16,3,224,224)
	#target_x = torch.rand(2,16,14,14,2*2*3*9)
	path = './test.mp4'
	mask_generator = CubeMaskGenerator(input_size=(8,14,14),min_num_patches=16)

	v_decoder = DecordInit()
	v_reader = v_decoder(path)
	# Sampling video frames
	total_frames = len(v_reader)
	align_transform = T.Compose([
		T.RandomResizedCrop(size=(224, 224), area_range=(0.9, 1.0), interpolation=3), #InterpolationMode.BICUBIC
		])
	mean = (0.485, 0.456, 0.406) 
	std = (0.229, 0.224, 0.225) 
	aug_transform = T.Compose([T.ToTensor(),T.Normalize(mean,std)])
	temporal_sample = T.TemporalRandomCrop(16*4)
	start_frame_ind, end_frame_ind = temporal_sample(total_frames)
	frame_indice = np.linspace(0, end_frame_ind-start_frame_ind-1, 
							   16, dtype=int)
	video = v_reader.get_batch(frame_indice).asnumpy()
	del v_reader
	
	# Video align transform: T C H W
	with torch.no_grad():
		video = torch.from_numpy(video).permute(0,3,1,2)
		align_transform.randomize_parameters()
		video = align_transform(video)
		unnorm_video = video
	label = np.stack(list(map(extract_hog_features, video.permute(0,2,3,1).numpy())), axis=0) # T H W C -> T H' W' C'
	#_, hog_image = hog(video.permute(0,2,3,1).numpy()[0][:,:,2], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2', feature_vector=False, visualize=True)
	mask, cube_marker = mask_generator() # T' H' W'

	video = aug_transform(video)
	label = torch.from_numpy(label)
	mask = torch.from_numpy(mask)
	video = video.to(device)
	label = label.to(device)
	mask = mask.to(device)
	print(video.shape, label.shape, mask.shape, cube_marker)
	
	# 3. mask hog prediction
	with torch.no_grad():
		out, loss, mask_preds, center_index = model(video.unsqueeze(0), label.unsqueeze(0), mask.unsqueeze(0), [cube_marker], visualize=True)
	print(out.shape, loss.item(), mask_preds.shape, center_index)
	# (1)visualize the de-norm hog prediction map of target frame
	eps=1e-5
	mask_preds = mask_preds[:,:,:,:,0,:][0][0].detach().cpu().numpy()
	real_norm = get_hog_norm(unnorm_video.permute(0,2,3,1).numpy()[center_index][0,:,:,0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
	print(real_norm.shape) # [28,28,1]
	mask_preds = mask_preds * real_norm
	save_path = f'./hog_pred.jpg'
	visualize_hog(mask_preds, save_path)

	# (2)visualize the original target frame
	save_path = f'./real_img.jpg'
	io.imsave(save_path, unnorm_video.permute(0,2,3,1).numpy()[center_index][0,:,:,0])
	
	# (3)visualize the target masked frame
	mask = repeat(mask, 't h w -> (t dt) (h dh) (w dw)', dt=2, dh=16, dw=16).detach().cpu() # [16, 224, 224]
	mask = 1 - mask
	img_mask = unnorm_video.permute(0,2,3,1)[center_index][0,:,:,0] * mask[center_index][0]
	save_path = f"./mask_img.jpg"
	io.imsave(save_path, img_mask.numpy())
