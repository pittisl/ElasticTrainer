# Introduction
This code repository stores program implementation for the accepted MobiSys 2023 paper "ElasticTrainer: Speeding Up On-Device Training with Runtime Elastic Tensor Selection". According to our paper, the code is intended to be run on embedded devices (e.g., Raspberry Pi and Nvidia Jetson TX2), but also applicable to workstations.

If you want to quickly review the core of our implementation, please see `profiler.py`, `elastic_training` in `train.py`, and `selection_solver_DP.py`.

**The current code has NOT been finalized yet. Stay tuned for camera-ready paper release**.

# Requirements
* Python 3.7+
* tensorflow 2
* tensorflow-datasets
* tensorflow-addons
* vit-keras
* tqdm

# Usage
Pick up NN models and datasets to run. Use `python main.py --help` and `python profiler.py --help` to see configurable parameters.

Supported NN architectures:
* ResNet50 -- `resnet50`
* VGG16 -- `vgg16`
* MobileNetV2 -- `mobilenetv2`
* Vision Transformer (16x16 patch) -- `vit`

Supported datasets:
* [CUB-200 (200 classes)](https://www.vision.caltech.edu/datasets/cub_200_2011/) -- `caltech_birds2011`
* [Oxford-IIIT pet (37 classes)](https://www.robots.ox.ac.uk/~vgg/data/pets/) -- `oxford_iiit_pet`
* [Stanford Dogs (120 classes)](http://vision.stanford.edu/aditya86/ImageNetDogs/) -- `stanford_dogs`

Below shows an example of training ResNet50 on CUB-200 dataset with our ElasticTrainer. First, profile the tensor timing on your dedicated device:
```
python profiler.py --model_name resnet50 \
                   --num_classes 200
```
Then start training your model on the device with speedup ratio of 0.5 (i.e., 2x faster):
```
python main.py --model_name resnet50 \
               --dataset_name caltech_birds2011 \
               --train_type elastic_training \
               --rho 0.5
```
