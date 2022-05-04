# PyTorch implementation of Video Transformer Benchmarks
This repository is mainly built upon [Pytorch](https://pytorch.org/) and [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/). We wish to maintain a collections of scalable video transformer benchmarks, and discuss the training recipes of how to train a big video transformer model.

Now, we implement the [TimeSformer](https://arxiv.org/abs/2102.05095), [ViViT](https://arxiv.org/abs/2103.15691) and [MaskFeat](https://arxiv.org/abs/2112.09133). And we have pre-trained the `TimeSformer-B`, `ViViT-B` and `MaskFeat` on [Kinetics400/600](https://deepmind.com/research/open-source/kinetics), but still can't guarantee the performance reported in the paper. However, we find some relevant hyper-parameters which may help us to reach the target performance.

## Update
1. We have fixed serval known issues and now can build script to pretrain `MViT-B` with `MaskFeat` or finetune `MViT-B`/`TimeSformer-B`/`ViViT-B` on K400. 
2. We have reimplemented the methods of hog extraction and hog prediction in [MaskFeat](https://arxiv.org/abs/2112.09133) which are currently more efficient to pretrain.
3. Note that if someone want to train `TimeSformer-B` or `ViViT-B` with current repo, they need to carefully adjust the learning rate and weight decay for a better performance. For example, you can can choose 0.005 for peak learning rate and 0.0001 for weight decay by default.

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
 * Initialize from the `ViT` pre-trained model can be found [here](https://drive.google.com/file/d/1QjGpbR8K4Cf4TJaDc60liVhBvPtrc2v4/view?usp=sharing).

## TODO
- [√] add more `TimeSformer` and `ViViT` variants pre-trained weights.
	- A larger version and other operation types.
- [√] add `linear prob` and `finetune recipe`.
	- Make available to transfer the pre-trained model to downstream task.
- [ ] add more scalable Video Transformer benchmarks.
	- We will mainly focus on the data-efficient models.
- [ ] add more robust objective functions.
	- Pre-train the model through the dominated self-supervised methods, e.g [Mask Image Modeling](https://arxiv.org/abs/2111.06377).

## Setup
```shell
pip install -r requirements.txt
```

## Usage
### Training
```shell
# path to Kinetics400 train set and val set
TRAIN_DATA_PATH='/path/to/Kinetics400/train_list.txt'
VAL_DATA_PATH='/path/to/Kinetics400/val_list.txt'
# path to root directory
ROOT_DIR='/path/to/work_space'
# path to pretrain weights
PRETRAIN_WEIGHTS='/path/to/weights'

# pretrain mvit using maskfeat
python model_pretrain.py \
	-lr 8e-4 -epoch 300 -batch_size 16 -num_workers 8 -frame_interval 4 -num_frames 16 -num_class 400 \
	-root_dir $ROOT_DIR -train_data_path $TRAIN_DATA_PATH

# finetune mvit with maskfeat pretrain weights
python model_pretrain.py \
	-lr 0.005 -epoch 200 -batch_size 8 -num_workers 4 -num_frames 16 -frame_interval 4 -num_class 400 \
	-arch 'mvit' -optim_type 'adamw' -lr_schedule 'cosine' -objective 'supervised' -mixup True \
	-auto_augment 'rand_aug' -root_dir $ROOT_DIR -train_data_path $TRAIN_DATA_PATH \
	-val_data_path $VAL_DATA_PATH -pretrain_pth $PRETRAIN_WEIGHTS

# finetune timesformer with imagenet pretrain weights
python model_pretrain.py \
	-lr 0.005 -epoch 30 -batch_size 8 -num_workers 4 -num_frames 8 -frame_interval 32 -num_class 400 \
	-arch 'timesformer' -attention_type 'divided_space_time' -optim_type 'sgd' -lr_schedule 'cosine' \
	-objective 'supervised' -root_dir $ROOT_DIR -train_data_path $TRAIN_DATA_PATH \
	-val_data_path $VAL_DATA_PATH -pretrain_pth $PRETRAIN_WEIGHTS -weights_from 'imagenet'

# finetune vivit with imagenet pretrain weights
python model_pretrain.py \
	-lr 0.005 -epoch 30 -batch_size 8 -num_workers 4 -num_frames 16 -frame_interval 16 -num_class 400 \
	-arch 'vivit' -attention_type 'fact_encoder' -optim_type 'sgd' -lr_schedule 'cosine' \
	-objective 'supervised' -root_dir $ROOT_DIR -train_data_path $TRAIN_DATA_PATH \
	-val_data_path $VAL_DATA_PATH -pretrain_pth $PRETRAIN_WEIGHTS -weights_from 'imagenet'

```
The minimal folder structure will look like as belows.
```
root_dir
├── results
│   ├── experiment_tag
│   │   ├── ckpt
│   │   ├── log
```

## Result
### Kinetics-400/600

#### 1. Model Zoo

| name | weights from | dataset | epochs | num frames | spatial crop | top1_acc | top5_acc | weight | log |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| TimeSformer-B | ImageNet-21K | K600 | 15e | 8 | 224 | 78.4 | 93.6 | [Google drive](https://drive.google.com/file/d/1-BSNROh35fiOIBcmtFNgWHEY_JC5UNDx/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1I5L41ZFHHSvFJttYt8F0Og)(code: yr4j) | [log](demo/log_arch_timesformer_lr5e-3_bs8_nw4_open.txt) |
| ViViT-B | ImageNet-21K | K400 | 30e | 16 | 224 | 75.2 | 91.5 | [Google drive](https://drive.google.com/file/d/1-JVhSN3QHKUOLkXLWXWn5drdvKn0gPll/view?usp=sharing) | |
| MaskFeat | from scratch | K400 | 100e | 16 | 224 | | | [Google drive](https://drive.google.com/file/d/1h3Q-267qV9kIcTT9Sct-zQzVvXljhyWW/view?usp=sharing) | |

#### 1.1 Visualize

For each column, we show the masked input(left), HOG predictions(middle) and original video frame(right).
<p align="center">
  <img src="https://user-images.githubusercontent.com/94091472/153032427-732c743d-aaca-4a3f-98ac-ae35b2cf6140.png" width="480">
</p>

Here, we show the extracted attention map of a random frame sampled from the demo video.
<p align="center">
  <img src="https://user-images.githubusercontent.com/94091472/157862844-2e380394-d491-415f-a366-7c657918d934.png" width="1500">
</p>

<br />

#### 2. Train Recipe(ablation study)
#### 2.1 Acc

| operation | top1_acc | top5_acc | top1_acc (three crop) |
|:----|:----:|:----:|:----:|
| base | 68.2 | 87.6 | - |
| + `frame_interval` 4 -> 16 (span more time) | 72.9(+4.7) | 91.0(+3.4) | - |
| + RandomCrop, flip (overcome overfit) | 75.7(+2.8) | 92.5(+1.5) | - |
| + `batch size` 16 -> 8 (more iterations) | 75.8(+0.1) | 92.4(-0.1) | - |
| + `frame_interval` 16 -> 24 (span more time) | 77.7(+1.9) | 93.3(+0.9) | 78.4 |
| + `frame_interval` 24 -> 32 (span more time) | 78.4(+0.7) | 94.0(+0.7) | 79.1 |

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
this repo is built on top of [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/), [pytorchvideo](https://github.com/facebookresearch/pytorchvideo/tree/9d0ca900f0427ed9b47b6182ad05f75c0e66274b), [skimage](https://github.com/scikit-image/scikit-image), [decord](https://github.com/dmlc/decord) and [kornia](https://github.com/kornia/kornia). I also learn many code designs from [MMaction2](https://github.com/open-mmlab/mmaction2). I thank the authors for releasing their code.

## Contribution
I look forward to seeing one can provide some ideas about the repo, please feel free to report it in the issue, or even better, submit a pull request.

And your star is my motivation, thank u~
