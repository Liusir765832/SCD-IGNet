# SCD-IGNet

This repository implements a PyTorch re-implementation of the research paper: "**SCD-IGNet: Enhanced Semantic Segmentation of Low-Light Rainy Images via Symmetric Cross-Decoupling and Illumination Guidance**"



## Dataset

The Nightcity-fine dataset is derived from the paper "Disentangle then Parse: Night-time Semantic Segmentation with Illumination Disentanglement" and can be downloaded via this[Google Drive link](https://drive.google.com/file/d/1Ilj99NMAmkZIPQcVOd6cJebnKXjJ-Sit/view?usp=drive_link).

The Nightcity-rain dataset was created by adding rain streaks to the Nightcity-fine dataset, and it's available for download through a [cloud storage link](). 

 Additionally, we provide MatLab code for generating rain streaks, enabling users to add rain effects to any dataset. You can download the rain streak dataset via the [cloud storage link](), and then use the MatLab code in the "rain" folder to generate the low-light rainy dataset NightCity-rain based on the NightCity-fine dataset.



## Environment Setup

Set up your environment with these steps:

```bash
conda create -n dtp python=3.10
conda activate scdig
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
# Alternatively: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
pip install tensorboard
pip install -U openmim
mim install mmcv-full
pip install -v -e .
# Alternatively: python setup.py develop
```

You can also install the required Python packages via the requirements.txt file.

```shell
pip install -r requirements.txt
```



## Preparation

### Download and Organize

1. Decompress the `nightcity-rain.zip` dataset and relocate it to `./data/nightcity-rain`.
2. Decompress the `nightcity-fine.zip` dataset and relocate it to `./data/nightcity-fine`.

```plaintext
.
├── checkpoints
│   ├── cfg
|   └── pth
|		 └── simmim_pretrain__swin_base__img192_window6__800ep.pth
├── custom
├── custom-tools
│   ├── dist_test.sh
│   ├── dist_train.sh
│   ├── test.py
│   └── train.py
├── data
│   └── nightcity-rain
│       ├── train
│       └── val
├── mmseg
├── readme.md
├── requirements.txt
├── setup.cfg
└── setup.py
```



## Training

1. Download pre-training weight from [Google Drive](https://drive.google.com/file/d/15zENvGjHlM71uKQ3d2FbljWPubtrPtjl/view).
2. Convert it to MMSeg format using:

```shell
python custom-tools/swin2mmseg.py </path/to/pretrain> checkpoints/simmim_pretrain__swin_base__img192_window6__800ep.pth
```

​    3. Start training with:

```shell
python custom-tools/train.py checkpoints/cfg/cfg.py
```



## Test

Execute tests using:

```shell
python custom-tools/test.py checkpoints/night/cfg.py checkpoint.pth --eval mIoU --aug-test
```



## Acknowledgements

This dataset is refined based on the dataset of [NightCity-fine](https://openaccess.thecvf.com//content/ICCV2023/papers/Wei_Disentangle_then_Parse_Night-time_Semantic_Segmentation_with_Illumination_Disentanglement_ICCV_2023_paper.pdf) by Zhixiang Wei *et al.* 

The method for adding rain streaks to images is based on the approach described in the paper **"Single Image Deraining using Scale-Aware Multi-Stage Recurrent Network"** [method](https://github.com/liruoteng/RainStreakGen.git).
