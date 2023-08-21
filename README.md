# GAN Compression
### [project](https://hanlab.mit.edu/projects/gancompression/) | [paper](https://arxiv.org/abs/2003.08936) | [videos](https://www.youtube.com/playlist?list=PL80kAHvQbh-r5R8UmXhQK1ndqRvPNw_ex) | [slides](https://hanlab.mit.edu/projects/gancompression/resources/546-slides.pdf) 

**[NEW!]** GAN Compression is accepted by T-PAMI! We released our T-PAMI version in the [arXiv v4](https://arxiv.org/abs/2003.08936)!

![teaser](imgs/teaser.png)
*We introduce GAN Compression, a general-purpose method for compressing conditional GANs. Our method reduces the computation of widely-used conditional GAN models, including pix2pix by 9-29x while preserving the visual fidelity. Our method is effective for a wide range of generator architectures, learning objectives, and both paired and unpaired settings.*

GAN Compression: Efficient Architectures for Interactive Conditional GANs<br>
[Muyang Li](https://lmxyy.me/), [Ji Lin](http://linji.me/), [Yaoyao Ding](https://yaoyaoding.com/), [Zhijian Liu](http://zhijianliu.com/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), and [Song Han](https://songhan.mit.edu/)<br>
MIT, Adobe Research, SJTU<br>
In CVPR 2020.  

## Demos
<p align="center">
  <img src="imgs/demo_xavier.gif" width=600>
</p>

## Overview

![overview](imgs/overview.png)*GAN Compression framework: ① Given a pre-trained teacher generator G', we distill a smaller “once-for-all” student generator G that contains all possible channel numbers through weight sharing. We choose different channel numbers for the student generator G at each training step. ② We then extract many sub-generators from the “once-for-all” generator and evaluate their performance. No retraining is needed, which is the advantage of the “once-for-all” generator. ③ Finally, we choose the best sub-generator given the compression ratio target and performance target (FID or mIoU) using either brute-force or evolutionary search method. Optionally, we perform additional fine-tuning, and obtain the final compressed model.*

## Performance

![performance](imgs/performance.jpeg)

*GAN Compression reduces the computation of pix2pix by 9-21x, and model size by 4.6-33x.*

## Colab Notebook

PyTorch Colab notebook: [pix2pix](https://colab.research.google.com/github/mit-han-lab/gan-compression/blob/master/pix2pix.ipynb).

## Prerequisites

* Linux
* Python 3
* CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation

- Clone this repo:

  ```shell
  git clone git@github.com:mit-han-lab/gan-compression.git
  cd gan-compression
  ```

- Install [PyTorch](https://pytorch.org) 1.4 and other dependencies (e.g., torchvision).

  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, we provide an installation script `scripts/conda_deps.sh`. Alternatively, you can create a new Conda environment using `conda env create -f environment.yml`.

### Pix2pix

#### Setup

* Download the pix2pix dataset (e.g., edges2shoes).

  ```shell
  bash datasets/download_pix2pix_dataset.sh edges2shoes-r
  ```

* Get the statistical information for the ground-truth images for your dataset to compute FID. We provide pre-prepared real statistics for several datasets. For example,

  ```shell
  bash datasets/download_real_stat.sh edges2shoes-r B
  bash datasets/download_real_stat.sh edges2shoes-r subtrain_B
  ```

#### Apply a Pre-trained Model

* Download the pre-trained models.

  ```shell
  python scripts/download_model.py --model pix2pix --task edges2shoes-r --stage full
  python scripts/download_model.py --model pix2pix --task edges2shoes-r --stage compressed
  ```

* Test the original full model.

  ```shell
  bash scripts/pix2pix/edges2shoes-r/test_full.sh
  ```

* Test the compressed model.

  ```shell
  bash scripts/pix2pix/edges2shoes-r/test_compressed.sh
  ```
  
* Measure the latency of the two models.

  ```shell
  bash scripts/pix2pix/edges2shoes-r/latency_full.sh
  bash scripts/pix2pix/edges2shoes-r/latency_compressed.sh
  ```

* There may be a little differences between the results of above models and those of the paper since we retrained the models. We also release the compressed models of the paper. If there are such inconsistencies, you could try the following commands to test our paper models:

  ```shell
  python scripts/download_model.py --model pix2pix --task edges2shoes-r --stage legacy
  bash scripts/pix2pix/edges2shoes-r/test_legacy.sh
  bash scripts/pix2pix/edges2shoes-r/latency_legacy.sh
  ```


### <span id="cityscapes">Cityscapes Dataset</span>

For the Cityscapes dataset, we cannot provide it due to license issue. Please download the dataset from https://cityscapes-dataset.com and use the script [prepare_cityscapes_dataset.py](datasets/prepare_cityscapes_dataset.py) to preprocess it. You need to download `gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip` and unzip them in the same folder. For example, you may put `gtFine` and `leftImg8bit` in `database/cityscapes-origin`. You need to prepare the dataset with the following commands:

```shell
python datasets/get_trainIds.py database/cityscapes-origin/gtFine/
python datasets/prepare_cityscapes_dataset.py \
--gtFine_dir database/cityscapes-origin/gtFine \
--leftImg8bit_dir database/cityscapes-origin/leftImg8bit \
--output_dir database/cityscapes \
--train_table_path datasets/train_table.txt \
--val_table_path datasets/val_table.txt
```

You will get a preprocessed dataset in `database/cityscapes` and a mapping table (used to compute mIoU) in `dataset/table.txt`.

To support mIoU computation, you need to download a pre-trained DRN model `drn-d-105_ms_cityscapes.pth` from http://go.yf.io/drn-cityscapes-models. By default, we put the drn model in the root directory of the repo. Then you can test our compressed models on cityscapes after you have downloaded our models.

### COCO-Stuff Dataset

We follow the same COCO-Stuff dataset preparation as [NVlabs/spade](https://github.com/NVlabs/SPADE). Specifically, you need to download `train2017.zip`, `val2017.zip`, `stuffthingmaps_trainval2017.zip`, and `annotations_trainval2017.zip` from [nightrome/cocostuff](https://github.com/nightrome/cocostuff). The images, labels, and instance maps should be arranged in the same directory structure as in [datasets/coco_stuff](https://github.com/NVlabs/SPADE/tree/master/datasets/coco_stuff). In particular, we used an instance map that combines both the  boundaries of "things instance map" and "stuff label map". To do this,  we used a simple script [datasets/coco_generate_instance_map.py](datasets/coco_generate_instance_map.py). 

To support mIoU computation, you need to download a pre-trained DeeplabV2 model [deeplabv2_resnet101_msc-cocostuff164k-100000.pth](https://github.com/kazuto1011/deeplab-pytorch/releases/download/v1.0/deeplabv2_resnet101_msc-cocostuff164k-100000.pth) and also put it in the root directory of the repo.

### Performance of Released Models

Here we show the performance of all our released models:

<table style="undefined;table-layout: fixed; width: 868px">
<colgroup>
<col style="width: 130px">
<col style="width: 130px">
<col style="width: 130px">
<col style="width: 130px">
<col style="width: 130px">
<col style="width: 109px">
<col style="width: 109px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Dataset</th>
    <th rowspan="2">Method</th>
    <th rowspan="2">#Parameters</th>
    <th rowspan="2">MACs</th>
    <th colspan="2">Metric</th>
  </tr>
  <tr>
    <td>FID</td>
    <td>mIoU</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>GAN Compression (Paper)</td>
    <td>0.342M</td>
    <td>2.67G</td>
    <td>65.33</td>
    <td>--</td>
  </tr>
  <tr>
    <td>GAN Compression (Retrained)</td>
    <td>0.357M</td>
    <td>2.55G</td>
    <td>65.12</td>
    <td>--</td>
  </tr>
  <tr>
    <td>Fast GAN Compression</td>
    <td>0.355M</td>
    <td>2.64G</td>
    <td>65.19</td>
    <td>--</td>
  </tr>
  <tr>
    <td rowspan="11">Pix2pix</td>
    <td rowspan="4">edges→shoes</td>
    <td>Original</td>
    <td>11.4M</td>
    <td>56.8G</td>
    <td>24.12</td>
    <td>--</td>
  </tr>
  <tr>
    <td>GAN Compression (Paper)</td>
    <td>0.700M</td>
    <td>4.81G</td>
    <td>26.60</td>
    <td>--</td>
  </tr>
  <tr>
    <td>GAN Compression (Retrained)</td>
    <td>0.822M</td>
    <td>4.99G</td>
    <td>26.70</td>
    <td>--</td>
  </tr>
  <tr>
    <td>Fast GAN Compression</td>
    <td>0.703M</td>
    <td>4.83G</td>
    <td>25.76</td>
    <td>--</td>
  </tr>
  <tr>
    <td rowspan="4">Cityscapes</td>
    <td>Original</td>
    <td>11.4M</td>
    <td>56.8G</td>
    <td>--</td>
    <td>42.06</td>
  </tr>
  <tr>
    <td>GAN Compression (Paper)</td>
    <td>0.707M</td>
    <td>5.66G</td>
    <td>--</td>
    <td>40.77</td>
  </tr>
  <tr>
    <td>GAN Compression (Retrained)</td>
    <td>0.781M</td>
    <td>5.59G</td>
    <td>--</td>
    <td>38.63</td>
  </tr>
  <tr>
    <td>Fast GAN Compression</td>
    <td>0.867M</td>
    <td>5.61G</td>
    <td>--</td>
    <td>41.71</td>
  </tr>
  <tr>
    <td rowspan="3">map→arial photo<br></td>
    <td>Original</td>
    <td>11.4M</td>
    <td>56.8G</td>
    <td>47.91</td>
    <td>--</td>
  </tr>
  <tr>
    <td>GAN Compression</td>
    <td>0.746M</td>
    <td>4.68G</td>
    <td>48.02</td>
    <td>--</td>
  </tr>
  <tr>
    <td>Fast GAN Compression</td>
    <td>0.708M</td>
    <td>4.53G</td>
    <td>48.67</td>
    <td>--</td>
  </tr>
  <tr>
    <td>GAN Compression (Paper)</td>
    <td>20.4M</td>
    <td>31.7G</td>
    <td>55.19</td>
    <td>61.22</td>
  </tr>
  <tr>
    <td>GAN Compression (Retrained)</td>
    <td>21.0M</td>
    <td>31.2G</td>
    <td>56.43</td>
    <td>60.29</td>
  </tr>
  <tr>
    <td>Fast GAN Compression</td>
    <td>20.2M</td>
    <td>31.3G</td>
    <td>56.25</td>
    <td>61.17</td>
  </tr>
  <tr>
    <td rowspan="2">COCO-Stuff</td>
    <td>Original</td>
    <td>97.5M</td>
    <td>191G</td>
    <td>21.38</td>
    <td>38.78</td>
  </tr>
  <tr>
    <td>Fast GAN Compression</td>
    <td>26.0M</td>
    <td>35.5G</td>
    <td>25.06</td>
    <td>35.05</td>
  </tr>
  <tr>
    <td>Fast GAN Compression</td>
    <td>1.10M</td>
    <td>2.63G</td>
    <td>30.53</td>
    <td>--</td>
  </tr>
</tbody>
</table>

### Training

Please refer to the tutorial of [Fast GAN Compression](docs/tutorials/fast_gan_compression.md) and [GAN Compression](docs/tutorials/gan_compression.md) on how to train models on our datasets and your own.

### FID Computation

To compute the FID score, you need to get some statistical information from the groud-truth images of your dataset. We provide a script [get_real_stat.py](./get_real_stat.py) to extract statistical information. For example, for the edges2shoes dataset, you could run the following command:

  ```shell
python get_real_stat.py \
--dataroot database/edges2shoes-r \
--output_path real_stat/edges2shoes-r_B.npz \
--direction AtoB
  ```

For paired image-to-image translation (pix2pix), we calculate the FID between generated test images to real test images.

## [Code Structure](docs/overview.md)

To help users better understand and use our code, we briefly overview the functionality and implementation of each package and each module.

## Citation

If you use this code for your research, please cite our [paper](https://arxiv.org/pdf/2003.08936).
```bibtex
@inproceedings{li2020gan,
  title={GAN Compression: Efficient Architectures for Interactive Conditional GANs},
  author={Li, Muyang and Lin, Ji and Ding, Yaoyao and Liu, Zhijian and Zhu, Jun-Yan and Han, Song},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```


## Acknowledgements

Our code is developed based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

We also thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID computation, [drn](https://github.com/fyu/drn) for cityscapes mIoU computation and [deeplabv2](https://github.com/kazuto1011/deeplab-pytorch) for Coco-Stuff mIoU computation.
