#!/bin/bash

# ResNet50, caltech_birds2011, ElasticTrainer
python3 profiler.py --model_name resnet50 --num_classes 200

python3 main.py --model_name resnet50 --dataset_name caltech_birds2011 --train_type elastic_training

# ResNet50, oxford_iiit_pet, ElasticTrainer
python3 profiler.py --model_name resnet50 --num_classes 37

python3 main.py --model_name resnet50 --dataset_name oxford_iiit_pet --train_type elastic_training

# ResNet50, stanford_dogs, ElasticTrainer
python3 profiler.py --model_name resnet50 --num_classes 120

python3 main.py --model_name resnet50 --dataset_name stanford_dogs --train_type elastic_training