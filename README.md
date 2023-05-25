# ElasticTrainer: Speeding Up On-Device Training with Runtime Elastic Tensor Selection (MobiSys'23)

## Introduction
This is the official code repository for our MobiSys 2023 paper "ElasticTrainer: Speeding Up On-Device Training with Runtime Elastic Tensor Selection". ElasticTrainer can speed up on-device NN training by ***adaptively training the minimal set of important parameters on the fly*** :rocket:, with user-defined speedup and without noticeable accuracy loss. According to our paper, the code is intended to be run on embedded devices (e.g., Raspberry Pi and Nvidia Jetson TX2), but it is also applicable to stronger platforms such as workstations. 

:mag_right: Looking for the core of our implementation? We suggest you take a look at the following:
* Tensor Timing Profiler -- [profiler.py](https://github.com/HelloKevin07/ElasticTrainer/blob/main/profiler.py)
* Tensor Importance Evaluator -- [elastic_training](https://github.com/HelloKevin07/ElasticTrainer/blob/main/train.py#L407) in [train.py](https://github.com/HelloKevin07/ElasticTrainer/blob/main/train.py)
* Tensor Selector by Dynamic Programming -- [selection_solver_DP.py](https://github.com/HelloKevin07/ElasticTrainer/blob/main/selection_solver_DP.py).

:chart_with_upwards_trend: Want to reproduce our paper results? Please check instructions [here](#reproducing-paper-results).

## License

Our source code is released under MIT License.

## Requirements
* Python 3.7+
* tensorflow 2
* tensorflow-datasets
* tensorflow-addons
* [tensorboard_plugin_profile](https://www.tensorflow.org/guide/profiler)
* [vit-keras](https://github.com/faustomorales/vit-keras)
* tqdm

The software versions are platform dependent. In general, installing the most recent versions should work for typical workstations. For Nvidia Jetson TX2 and Raspberry Pi 4B, please check our provided OS images described [here](https://github.com/HelloKevin07/ElasticTrainer#reproducing-paper-results).

## General Usage
Select NN models and datasets to run. Use `python main.py --help` and `python profiler.py --help` to see configurable parameters. The NN architectures and datasets should be downloaded automatically. We use [tensorflow-datasets](https://www.tensorflow.org/datasets/api_docs/python/tfds) API to download datasets from tensorflow's [dataset list](https://www.tensorflow.org/datasets/catalog/overview#all_datasets). If you encounter any errors (e.g., checksum error), please refer to [this instruction](https://www.tensorflow.org/datasets/overview#manual_download_if_download_fails) for manually downloading.

Supported NN architectures:
* ResNet50 -- `resnet50`
* VGG16 -- `vgg16`
* MobileNetV2 -- `mobilenetv2`
* Vision Transformer (16x16 patch) -- `vit`

Supported datasets:
* [CUB-200 (200 classes)](https://www.vision.caltech.edu/datasets/cub_200_2011/) -- `caltech_birds2011`
* [Oxford-IIIT Pet (37 classes)](https://www.robots.ox.ac.uk/~vgg/data/pets/) -- `oxford_iiit_pet`
* [Stanford Dogs (120 classes)](http://vision.stanford.edu/aditya86/ImageNetDogs/) -- `stanford_dogs`

## Running an Example

Below shows an example of training ResNet50 on CUB-200 dataset with our ElasticTrainer. First, profile the tensor timing on your dedicated device:
```
python profiler.py --model_name resnet50 \
                   --num_classes 200
```
Then start training your model on the device with speedup ratio of 0.5 (i.e., 2x faster in wall time):
```
python main.py --model_name resnet50 \
               --dataset_name caltech_birds2011 \
               --train_type elastic_training \
               --rho 0.5
```
Please note that the wall training time should exclude validation time.

## FAQs
**Q1: Why tensorflow, not pytorch?**

We are aware that pytorch is a dominant NN library in AI research community. However, to the best of our knowledge, pytorch's profiler is incapable of presenting structured timing information of ops in backward pass, and some of the provided measurements are not even reliable. Check [link1](https://github.com/pytorch/kineto/issues/580), [link2](https://github.com/pytorch/kineto/pull/372), and [link3](https://github.com/pytorch/pytorch/issues/30711) for details. In comparison, tensorflow profiler provides accurate, well-structured, and human-readable timing information for us to parse and group, but tensorflow's profiler only works for tensorflow models and codes. If you insist on working with pytorch, I can only suggest you use FLOPs instead of wall-clock time as the tensor timing metric. That means you need to write your own code to derive FLOPs and it basically cannot reflect actual speed-up.

**Q2: ElasticTrainer VS. Parameter-Efficient Fine-Tuning (PEFT) for recent Large Language Models (LLMs)?**

If you are an NLP expert, you may know there are many existing PEFT works in NLP area, such as [prompt tuning](https://arxiv.org/abs/2104.08691), [prefix tuning](https://arxiv.org/abs/2101.00190), and [LoRA](https://arxiv.org/abs/2106.09685). These works focus on minimizing the number trainable parameters (usually to <1%) because they speculate that variance rather than bias is a dominant factor in model generalization. 

However, **solely minimizing the number of trainable parameters doesn't gurantee wall-time speedup**. For example, prompt tuning still requires error gradients to propagate through the entire network, which leads to very limited wall-time speedup. On the other hand, nobody can promise variance is always a dominant factor in model generalzation. Unless you want to use super super large pretrained LLMs (e.g., GPT-3) with stunning zero-shot adaptability, applying PEFT to most medium-sized pretrained models would kill a lot of representational power for **complex generative tasks** (e.g., text summarization and math Q&A) and lose much accuracy.

**Q3: How are you able to select a subset of parameters to train?**

In tensorflow, `model.trainable_weights` gives you a list of all the trainable parameters. You can extract wanted ones into another list, say `var_list`. Then pass `var_list` to the optimizer, i.e., `optimizer.apply_gradients(zip(gradients, var_list))`. This process can be done at runtime but may cause frequent retracing in tensorflow. So you may need to manually free old graphs to avoid increasing memory usage, which is what we implemented originally. I later realized that maybe a better way to suppress retracing is to configure the [tf.function](https://www.tensorflow.org/api_docs/python/tf/function) decorator:

```python
@tf.function(
    experimental_relax_shapes=True,
    experimental_follow_type_hints=True,
)
def train_step(...)

# alternatively
@tf.function(reduce_retracing=True)
def train_step(...)
```

**Q4: Why are some tensors' timings not counted in our Tensor Timing Profiler?**

Because we cannot find related timings for these tensors from tensorflow's profiling results. That is, even for tensorflow profiler, it may fail to capture a few NN ops during profiling for no reason. We have no solution for that. One workaround can be using known op's timings to estimate missing op's timings based on their FLOPs relationships.

**Q5: What's the meaning of `(rho - 1/3)*3/2` in `elastic_training` in `train.py`?**

It converts training speedup to backward speedup based on the 2:1 FLOPs relationship between backward pass and forward pass. We did so to bypass profiling the forward time. Please note this is only an approximation, and we did this due to tight schedule when we rushing for this paper. To ensure precision, we highly recommend you do profile the forward time `T_fp` and backward time `T_bp`, and use `rho * (1 + T_fp/T_bp) - T_fp/T_bp` to for such conversion.

**Q6: Why is `rho` multiplied by `disco` in `elastic_training` in `train.py`**

`disco`, which is obtained [here](https://github.com/HelloKevin07/ElasticTrainer/blob/c9e53006f0ad64ca8392130b169952ff3c1cc57b/train.py#LL439C5-L439C72), is a heuristic factor that scales the `rho` a bit, to ensure the desired speedup can be achieved even if `t_dy` and `t_dw` lose much resolution after downscaling. The downside of `disco` is that sometimes it just becomes too small, and suppresses too much of the parameter selection. In that case, you can feel free to try removing this factor.

## Reproducing Paper Results
Please download our artifacts on Zenodo [link1](https://doi.org/10.5281/zenodo.7812218) and [link2](https://doi.org/10.5281/zenodo.7812233), and follow the detailed instructions in our [artifact appendix](docs/ElasticTrainer_AE_Appendix.pdf).
We provide experimental workflows that allow people to reproduce our main results in the paper. However, running all the experiments could take extremely long time (~800 hours), and thus we mark each experiment with its estimated execution time for users to choose based on their time budget. After you finish running each script, the figure will be automatically generated under `figures/`. For Nvidia Jetson TX2, we run experiments with its text-only interface, and to view the figures, you will need to switch back to the graphic interface.

We first describe how you can prepare the environment that allows you to run our experiments, and then we list command lines to reproduce every figure in our main results.

### Preparing Nvidia Jetson TX2
1. According to our artifact appendix, flash the Jetson using our provided OS image. Insert SSD.
2. Login the system where both username and password are `nvidia`. 
3. Run the following commands to finish preparation:
```
sudo su -
cd ~/src/ElasticTrainer
chmod +x *.sh
```

### Preparing Raspberry Pi 4B
1. Flash the Raspberry Pi using our provided OS image.
2. Open a terminal and run the following commands to finish preparation:
```
cd ~/src/ElasticTrainer
. ../kai_stl_code/venv/bin/activate
chmod +x *.sh
```

### Figure 15(a)(d) - A minimal reproduction of main results (~10 hours)
On Nvidia Jetson TX2:
```
./run_figure15ad.sh
```
### Figure 15 from (a) to (f) (~33 hours)
On Nvidia Jetson TX2:
```
./run_figure15.sh
```
Alternatively, if you want to exclude baseline schemes, run the following (~6.5 hours):
```
./run_figure15_ego.sh
```
### Figure 16 from (a) to (d) (~221 hours)
On Raspberry Pi 4B:
```
./run_figure16.sh
```
Alternatively, if you want to exclude baseline schemes, run the following (~52 hours):
```
./run_figure16_ego.sh
```
### Figure 17 (a)(c) (~15+190 hours)
Run the following command on both Nvidia Jetson TX2 (~15 hours) and Raspberry Pi 4B (~190 hours):
```
./run_figure17ac.sh
```
Alternatively, if you want to exclude baseline schemes, run the following command on both Nvidia Jetson TX2 (~9 hours) and Raspberry Pi 4B (~85 hours):
```
./run_figure17ac_ego.sh
```
### Figure 19 from (a) to (d) (~20+310 hours)
Run the following command on both Nvidia Jetson TX2 (~20 hours) and Raspberry Pi 4B (~310 hours):
```
./run_figure19.sh
```
Alternatively, if you want to exclude baseline schemes, run the following command on both Nvidia Jetson TX2 (~3.5 hours) and Raspberry Pi 4B (~50 hours):
```
./run_figure19_ego.sh
```

### Checking Results
All the experiment results should be generated under `figures/`. On Pi, directly click them to view. On Jetson, to check experiments results, you will need to switch to graphic mode:

```
sudo systemctl start graphical.target
``` 
In graphic mode, open a terminal, gain root privilege, and navigate to our code directory:
```
sudo su -
cd ~/src/ElasticTrainer
```
 All the figures are stored under `figures/`. Use `ls` command to check their file names. Use `evince` command to view the figures, for example, `evince xxx.pdf`. To go back to text-only mode, simply reboot the system. If you encounter any display issues, you can alternatively use tensorboard to view results. To enable tensorboard:
 ```
tensorboard --logdir logs
 ```
 Open Chrome/Chromium browser and visit URL http://localhost:6006/. On the right sidebar, make sure you switch from "Step" to "Relative" on "Settings->General".
