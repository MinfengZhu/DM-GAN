# *DM-GAN: Dynamic Memory Generative Adversarial Networks for Text-to-Image Synthesis*

### Introduction
This project page provides pytorch code that implements the following CVPR2019 paper:

**Title:** "DM-GAN: Dynamic Memory Generative Adversarial Networks for Text-to-Image Synthesis"

**Arxiv:** https://arxiv.org/abs/1904.01310

### How to use

**Python**

- Python2.7
- Pytorch0.4 (`conda install pytorch=0.4.1 cuda90 torchvision=0.2.1 -c pytorch`)
- tensorflow (`pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.12.0-cp27-none-linux_x86_64.whl`)
- `pip install easydict pathlib`
- `conda install requests nltk pandas scikit-image pyyaml cudatoolkit=9.0`


**Data**
1. Download metadata for [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) [coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9) and save them to `data/`
    - `python google_drive.py 1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ ./data/bird.zip`
    - `python google_drive.py 1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9 ./data/coco.zip`

2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
    - `cd data/birds`
    - `wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz`
    - `tar -xvzf CUB_200_2011.tgz`
    
3. Download [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`
    - `cd data/coco`
    - `wget http://images.cocodataset.org/zips/train2014.zip`
    - `wget http://images.cocodataset.org/zips/val2014.zip`
    - `unzip train2014.zip`
    - `unzip val2014.zip`
    - `mv train2014 images`
    - `cp val2014/* images`

**Pretrained Model**
- [DAMSM for bird](https://drive.google.com/open?id=1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V). Download and save it to `DAMSMencoders/`
    - `python google_drive.py 1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V DAMSMencoders/bird.zip`

- [DAMSM for coco](https://drive.google.com/open?id=1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ). Download and save it to `DAMSMencoders/`
    - `python google_drive.py 1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ DAMSMencoders/coco.zip`

- [DM-GAN for bird](https://drive.google.com/file/d/1j3IEcLy2LAWux9QRyjioOntKR2Vrwc5-). Download and save it to `models`
    - `python google_drive.py 1j3IEcLy2LAWux9QRyjioOntKR2Vrwc5- models/bird_DMGAN.pth`
- [DM-GAN for coco](https://drive.google.com/file/d/12IPwSL4w6Dp_b9ELnJDNXX9ijRcqHrV8). Download and save it to `models`
    - `python google_drive.py 12IPwSL4w6Dp_b9ELnJDNXX9ijRcqHrV8 models/coco_DMGAN.pth`
- [IS for bird](https://drive.google.com/file/d/0B3y_msrWZaXLMzNMNWhWdW0zVWs)
    - `python google_drive.py 0B3y_msrWZaXLMzNMNWhWdW0zVWs eval/IS/bird/inception_finetuned_models.zip`
- [FID for bird](https://drive.google.com/file/d/1fVc2_gnUE05SI7t2KaYGC-Elt0Cv5N7L)
    - `python google_drive.py 1fVc2_gnUE05SI7t2KaYGC-Elt0Cv5N7L eval/FID/bird_val.npz`
- [FID for coco](https://drive.google.com/file/d/1BHl_GyFyYr5qy78A2VKjIED1Hm64ZkDf)
    - `python google_drive.py 1BHl_GyFyYr5qy78A2VKjIED1Hm64ZkDf eval/FID/coco_val.npz`

**Training**
- bird: `python main.py --cfg cfg/bird_DMGAN.yml --gpu 0`
- coco: `python main.py --cfg cfg/coco_DMGAN.yml --gpu 0`

**Validation**
1. Images generation: 
    - `python main.py --cfg cfg/eval_bird.yml --gpu 0`
    - `python main.py --cfg cfg/eval_coco.yml --gpu 0`
2. Inception score ([IS for bird](https://github.com/hanzhanggit/StackGAN-inception-model), [IS for coco](https://github.com/openai/improved-gan/tree/master/inception_score)):
    - `CUDA_VISIBLE_DEVICES=0 python inception_score_bird.py --image_folder ../../../models/bird_DMGAN`
    - `CUDA_VISIBLE_DEVICES=1 python inception_score_coco.py ../../../models/coco_DMGAN`
3. FID ([Pytorch FID](https://github.com/mseitzer/pytorch-fid), [TF FID](https://github.com/bioinf-jku/TTUR)):
    - `python fid_score.py --gpu 0 --batch-size 50 --path1 bird_val.npz --path2 ../../models/bird_DMGAN`
    - `python fid_score.py --gpu 0 --batch-size 50 --path1 coco_val.npz --path2 ../../models/coco_DMGAN`

**Performance**

Note that after cleaning and refactoring the code of the paper, the results are slightly different.

|Model |R-precision↑  |IS↑  |FID↓ |
|----|-----| -----|---|
| bird_DMGAN (paper) | 72.31% ± 0.91%| 4.75 ± 0.07| 16.09|
| bird_DMGAN (Pretrained Model)| 74.48% ± 0.61% | 4.71 ± 0.06  |15.34|
| coco_DMGAN (paper) | 88.56% ± 0.28%| 30.49 ± 0.57 | 32.64|
| coco_DMGAN (Pretrained Model)| 89.52% ± 0.61%| 32.43 ± 0.58| 26.55|

### License
This code is released under the MIT License (refer to the LICENSE file for details). 
