# Introduction
This code repository stores program implementation for the accepted MobiSys 2023 paper "ElasticTrainer: Speeding Up On-Device Training with Runtime Elastic Tensor Selection". According to our paper, the code is intended to be run on embedded devices (e.g., Raspberry Pi and Nvidia Jetson TX2), but also applicable to workstations.

If you are looking for **the core of our implementation**, we suggest you take a look at the following:
* Tensor Timing Profiler -- `profiler.py`
* Tensor Importance Evaluator -- `elastic_training` in `train.py`
* Tensor Selector by Dynamic Programming -- `selection_solver_DP.py`.

**We are still finalizing our code. Please also stay tuned for camera-ready release of our paper**.

# Requirements
* Python 3.7+
* tensorflow 2
* tensorflow-datasets
* tensorflow-addons
* [tensorboard_plugin_profile](https://www.tensorflow.org/guide/profiler)
* [vit-keras](https://github.com/faustomorales/vit-keras)
* tqdm

# General Usage
Pick up NN models and datasets to run. Use `python main.py --help` and `python profiler.py --help` to see configurable parameters.

Supported NN architectures:
* ResNet50 -- `resnet50`
* VGG16 -- `vgg16`
* MobileNetV2 -- `mobilenetv2`
* Vision Transformer (16x16 patch) -- `vit`

Supported datasets:
* [CUB-200 (200 classes)](https://www.vision.caltech.edu/datasets/cub_200_2011/) -- `caltech_birds2011`
* [Oxford-IIIT Pet (37 classes)](https://www.robots.ox.ac.uk/~vgg/data/pets/) -- `oxford_iiit_pet`
* [Stanford Dogs (120 classes)](http://vision.stanford.edu/aditya86/ImageNetDogs/) -- `stanford_dogs`

**Note**: The NN architectures and datasets should be downloaded automatically. We use `tensorflow-datasets` APIs to download datasets from their collection list. If you encounter errors (e.g., checksum error) when downloading datasets, please refer to [instructions](https://www.tensorflow.org/datasets/overview#manual_download_if_download_fails) for manually downloading (not too difficult).

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
# Artifact Evaluation
We provide experimental workflows that allow people to reproduce our main results (including baselines' results) in the paper. However, running all the experiments could take extremely long time (~800 hours), and thus we list each experiment with its estimated running time for users to choose based on their time budget. After you finish running each script, the figure will be automatically generated under `figures/`. For Nvidia Jetson TX2, we run experiments with its text-only interface, and to view the figures, you will need to switch back to the graphic interface.

## Figure 15(a)(d) - A minimal reproduction (~10 hours)
On Nvidia Jetson TX2:
```
bash run_figure15ad.sh
```
## Figure 15 from (a) to (f) (~33 hours)
On Nvidia Jetson TX2:
```
bash run_figure15.sh
```
Alternatively, if you want to exclude baseline schemes, run the following (~6.5 hours):
```
bash run_figure15_ego.sh
```
## Figure 16 from (a) to (d) (~221 hours)
On Raspberry Pi 4B:
```
bash run_figure16.sh
```
Alternatively, if you want to exclude baseline schemes, run the following (~52 hours):
```
bash run_figure16_ego.sh
```
## Figure 17 (a)(c) (~15+190 hours)
Run the following command on both Nvidia Jetson TX2 (~15 hours) and Raspberry Pi 4B (~190 hours):
```
bash run_figure17ac.sh
```
Alternatively, if you want to exclude baseline schemes, run the following command on both Nvidia Jetson TX2 (~9 hours) and Raspberry Pi 4B (~85 hours):
```
bash run_figure17ac_ego.sh
```
## Figure 19 from (a) to (d) (~20+310 hours)
Run the following command on both Nvidia Jetson TX2 (~20 hours) and Raspberry Pi 4B (~310 hours):
```
bash run_figure19.sh
```
Alternatively, if you want to exclude baseline schemes, run the following command on both Nvidia Jetson TX2 (~3.5 hours) and Raspberry Pi 4B (~50 hours):
```
bash run_figure19_ego.sh
```
