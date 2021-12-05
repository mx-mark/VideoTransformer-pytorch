# PyTorch implementation of Video Transformer Benchmarks
This repository is mainly built upon [Pytorch](https://pytorch.org/) and [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/). We wish to maintain a collections of scalable video transformer benchmarks, and discuss the training recipes of how to train a big video transformer model.

Now, we implement the [TimeSformer](https://arxiv.org/abs/2102.05095) and [ViViT](https://arxiv.org/abs/2103.15691). And we have pre-trained the `TimeSformer-B` on [Kinetics600](https://deepmind.com/research/open-source/kinetics), but still can't guarantee the performance reported in the paper. However, we find some relevant hyper-parameters which may help us to reach the target performance.

## Table of Contents
1. [Difference](#difference)
2. [TODO](#todo)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Result](#result)
6. [Acknowledge](#acknowledge)
7. [Contribution](#contribution)

## Difference
In order to share the basic divided spatial-temporal attention module to different video transformer, we make some changes in the following apart.

### 1. Position embedding

We split the `position embedding` from *R(n<sup>t\*h\*w</sup>×d)* mentioned in the [ViViT](https://arxiv.org/abs/2103.15691) paper into *R(n<sup>h\*w</sup>×d)*
and *R(n<sup>t</sup>×d)* to stay the same as [TimeSformer](https://arxiv.org/abs/2102.05095).

### 2. Class token

In order to make clear whether to add the `class_token` into the module forward computation, we only compute the interaction between `class_token` and `query` when the current layer is the last layer (except `FFN`) of each transformer block.

### 3. Initialize from the pre-trained model

* Tokenization:  the token embedding filter can be chosen either `Conv2D` or `Conv3D`, and the initializing weights of `Conv3D` filters from `Conv2D` can be replicated along temporal dimension and averaging them or initialized with zeros along the temporal positions except at the center `t/2`.
 * Temporal `MSA` module weights: one can choose to copy the weights from spatial `MSA` module or initialize all weights with zeros.
 * Initialize from the `MAE` pre-trained model provided by [ZhiLiang](https://github.com/pengzhiliang/MAE-pytorch), where the class_token that does not appear in the `MAE` pre-train model is initialized from truncated normal distribution.

## TODO
- [ ] add more `TimeSformer` and `ViViT` variants pre-trained weights.
	- A larger version and other operation types.
- [ ] add `linear prob` and `partial fine-tune`.
	- Make available to transfer the pre-trained model to downstream task.
- [ ] add more scalable Video Transformer benchmarks.
	- We will also extend to multi-modality version, e.g [Perceiver](https://arxiv.org/abs/2107.14795) is coming soon.
- [ ] add more diverse objective functions.
	- Pre-train on larger dataset through the dominated self-supervised methods, e.g Contrastive Learning and [MAE](https://arxiv.org/abs/2111.06377).

## Setup
```bash
pip install -r requirements.txt
```

## Usage
### Training
```bash
# path to Kinetics600 train set
TRAIN_DATA_PATH='/path/to/Kinetics600/train_list.txt'
# path to root directory
ROOT_DIR='/path/to/work_space'

python model_pretrain.py \
	-lr 0.005 \
	-pretrain 'vit' \
	-epoch 15 \
	-batch_size 8 \
	-num_class 600 \
	-frame_interval 32 \
	-root_dir ROOT_DIR \
	-train_data_path TRAIN_DATA_PATH
```
The minimal folder structure will look like as belows.
```
root_dir
├── pretrain_model
│   ├── pretrain_mae_vit_base_mask_0.75_400e.pth
│   ├── vit_base_patch16_224.pth
├── results
│   ├── experiment_tag
│   │   ├── ckpt
│   │   ├── log
```

### Inference
```bash
# path to Kinetics600 pre-trained model
PRETRAIN_PATH='/path/to/pre-trained model'
# path to the test video sample
VIDEO_PATH='/path/to/video sample'

python model_inference.py \
	-pretrain PRETRAIN_PATH \
	-video_path VIDEO_PATH \
	-num_frames 8 \
	-frame_interval 32 \
```

## Result
### Kinetics-600

#### 1. Model Zoo

| name | pretrain | epochs | num frames | spatial crop | top1_acc | top5_acc | weight | log |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| TimeSformer-B | ImageNet-21K | 15e | 8 | 224 | 78.4 | 93.6 | Google drive or BaiduYun(code: mae6) | log |
<br />

#### 2. Train Recipe(ablation study)
#### 2.1 Acc

| operation | top1_acc | top5_acc | top1_acc (three crop) |
|:----|:----:|:----:|:----:|
| base | 68 | 78 | - |
| + `frame_interval` 4 -> 16 (span more time) | 73(+5) | 93(+) | - |
| + RandomCrop, flip (overcome overfit) | 75.7(+) | 80.5(+) | - |
| + `batch size` 16 -> 8 (more iterations) | 75.8 | 81.5 | - |
| + `frame_interval` 16 -> 24 (span more time) | 77.7(+) | 89 | 78.4 |
| + `frame_interval` 24 -> 32 (span more time) | 78.4(+0.7) | 93(+4) | 79.1 |

tips: `frame_interval` and `data augment` counts for the validation accuracy.

<br />

#### 2.2 Time

| operation | epoch_time |
|:----|:----:|
| base (start with DDP) | 9h+ |
| + `speed up training recipes` | 1h+ |
| + switch from `get_batch first` to `sample_Indice first` | 0.5h |
| + `batch size` 16 -> 8 | 33.32m |
| + `num_workers` 8 -> 4 | 35.52m |
| + `frame_interval` 16 -> 24 | 44.35m |

tips: Improve the `frame_interval` will drop a lot on time performance.

1.`speed up training recipes`: 
  * More GPU device.
  * `pin_memory=True`.
  * Avoid CPU->GPU Device transfer (such as `.item()`, `.numpy()`, `.cpu()` operations on tensor or `log` to disk).

2.`get_batch first` means that we firstly read all frames through the video reader, and then get the target slice of frames, so it largely slow down the data-loading speed.

<br />


## Acknowledge
this repo is built on top of [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/), [decord](https://github.com/dmlc/decord) and [kornia](https://github.com/kornia/kornia). I also learn many code designs from [MMaction2](https://github.com/open-mmlab/mmaction2). I thank the authors for releasing their code.

## Contribution
I look forward to seeing one can provide some ideas about the repo, please feel free to report it in the issue, or even better, submit a pull request.

And your star is my motivation, thank u~
